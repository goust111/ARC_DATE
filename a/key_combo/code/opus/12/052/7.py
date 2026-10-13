from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, paste


BACKGROUND = 0
QUADRANT_ORDER = ("TL", "TR", "BL", "BR")
QUADRANT_OFFSETS = {
    "TL": (0, 0),
    "TR": (0, 1),
    "BL": (1, 0),
    "BR": (1, 1),
}
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

RNG = np.random.default_rng()

QUADRANT_MIN = 7
QUADRANT_MAX = 10
SHAPE_MIN_AREA = 3
SHAPE_MAX_AREA = 6
SHAPE_SPAN_MAX = 4
MAX_GENERATION_ATTEMPTS = 400


def _mask_from_coords(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _canonical_shape_key(mask: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    base = np.asarray(mask, dtype=np.uint8)
    best = (base.tobytes(), tuple(map(int, base.shape)))
    for flipped in (base, np.fliplr(base)):
        for k in range(4):
            variant = np.rot90(flipped, k)
            key = (variant.tobytes(), tuple(map(int, variant.shape)))
            if key < best:
                best = key
    return best


def _extract_quadrant_objects(region: np.ndarray, quadrant: str) -> list[dict]:
    comps = find_components(
        region,
        bg=BACKGROUND,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    out: list[dict] = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(region, (r0, c0, r1, c1))
        mask = patch != BACKGROUND
        out.append(
            {
                "quadrant": quadrant,
                "bbox": (r0, c0, r1, c1),
                "r0": r0,
                "c0": c0,
                "color": int(comp["color"]),
                "mask": mask,
                "shape_key": _canonical_shape_key(mask),
            }
        )
    return out


def _ordered_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    assert height == width and height % 2 == 0, "expected an even square grid"
    n = height // 2
    all_objects: list[dict] = []
    for quadrant in QUADRANT_ORDER:
        qr, qc = QUADRANT_OFFSETS[quadrant]
        region = g[qr * n : (qr + 1) * n, qc * n : (qc + 1) * n]
        objs = _extract_quadrant_objects(region, quadrant)
        objs.sort(key=lambda obj: (obj["r0"], obj["c0"], obj["bbox"][2], obj["bbox"][3], obj["color"]))
        all_objects.extend(objs)
    return n, all_objects


def _shape_quadrant_counts(objects: list[dict]) -> dict[tuple[bytes, tuple[int, int]], int]:
    quadrants_by_shape: dict[tuple[bytes, tuple[int, int]], set[str]] = defaultdict(set)
    for obj in objects:
        quadrants_by_shape[obj["shape_key"]].add(obj["quadrant"])
    return {key: len(quadrants) for key, quadrants in quadrants_by_shape.items()}


def _render_instances(instances: list[dict], n: int, shape_counts: dict | None = None, *, force: str | None = None) -> np.ndarray:
    if shape_counts is None:
        shape_counts = _shape_quadrant_counts(instances)
    out = np.zeros((n, n), dtype=int)
    for quadrant in QUADRANT_ORDER:
        qobjs = [obj for obj in instances if obj["quadrant"] == quadrant]
        qobjs.sort(key=lambda obj: (obj["r0"], obj["c0"], obj["mask"].shape[0], obj["mask"].shape[1], obj["color"]))
        for obj in qobjs:
            patch = np.zeros(obj["mask"].shape, dtype=int)
            patch[obj["mask"]] = int(obj["color"])
            is_shared = bool(shape_counts[obj["shape_key"]] >= 2)
            if force == "opaque":
                is_shared = True
            elif force == "transparent":
                is_shared = False
            paste(out, patch, int(obj["r0"]), int(obj["c0"]), transparent=BACKGROUND, overwrite=is_shared)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the four quadrants onto one canvas, using cross-quadrant shape frequency to choose opaque or transparent stamping.

    Input structure:
    - The grid is an even square split into four equal square quadrants on a black background.
    - Each quadrant contains several single-color 4-connected objects.
    - Objects are compared by their tight binary masks, up to rotation and mirror flip.

    Rule:
    - Extract every object inside each quadrant and canonicalize its shape over all 8 rotation/flip variants.
    - Count in how many distinct quadrants each canonical shape appears.
    - Shapes seen in at least two quadrants are shared and stamp opaquely; shapes seen in exactly one quadrant are unique and stamp transparently.
    - Build one half-size output canvas and process quadrants in order TL, TR, BL, BR, using each object's local quadrant coordinates as output coordinates.

    Key insight:
    - The same shape-matching step that identifies shared sprites also controls rendering mode, so the final colors depend on both frequency grouping and layer order.
    """
    n, objects = _ordered_objects(grid)
    shape_counts = _shape_quadrant_counts(objects)
    return _render_instances(objects, n, shape_counts)


def _coords_from_mask(mask: np.ndarray, r0: int = 0, c0: int = 0) -> set[tuple[int, int]]:
    return {(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(mask)}


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].astype(bool)


def _random_connected_mask() -> tuple[np.ndarray, tuple[bytes, tuple[int, int]]]:
    for _ in range(200):
        h = int(RNG.integers(2, SHAPE_SPAN_MAX + 1))
        w = int(RNG.integers(2, SHAPE_SPAN_MAX + 1))
        max_area = min(SHAPE_MAX_AREA, h * w)
        if max_area < SHAPE_MIN_AREA:
            continue
        area = int(RNG.integers(SHAPE_MIN_AREA, max_area + 1))
        mask = np.zeros((h, w), dtype=bool)
        seed = (int(RNG.integers(0, h)), int(RNG.integers(0, w)))
        mask[seed] = True
        while int(mask.sum()) < area:
            frontier = set()
            for r, c in np.argwhere(mask):
                for dr, dc in NEIGHBORS4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = tuple(frontier)[int(RNG.integers(0, len(frontier)))]
            mask[nr, nc] = True
        if int(mask.sum()) != area:
            continue
        mask = _trim_mask(mask)
        key = _canonical_shape_key(mask)
        return mask, key
    raise RuntimeError("failed to build a connected mask")


def _fresh_mask(forbidden: set[tuple[bytes, tuple[int, int]]]) -> tuple[np.ndarray, tuple[bytes, tuple[int, int]]]:
    for _ in range(400):
        mask, key = _random_connected_mask()
        if key not in forbidden:
            return mask, key
    raise RuntimeError("failed to sample a fresh canonical shape")


def _random_variant(mask: np.ndarray) -> np.ndarray:
    variant = np.rot90(mask, int(RNG.integers(0, 4)))
    if bool(RNG.integers(0, 2)):
        variant = np.fliplr(variant)
    return np.asarray(variant, dtype=bool)


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _fits(mask: np.ndarray, n: int, r0: int, c0: int) -> bool:
    h, w = map(int, mask.shape)
    return 0 <= r0 and 0 <= c0 and r0 + h <= n and c0 + w <= n


def _place_on(mask: np.ndarray, target: np.ndarray, r0: int, c0: int) -> None:
    h, w = map(int, mask.shape)
    target[r0 : r0 + h, c0 : c0 + w] |= mask


def _find_position(
    mask: np.ndarray,
    n: int,
    local_occupied: np.ndarray,
    reserved_global: np.ndarray,
    *,
    tries: int = 200,
) -> tuple[int, int] | None:
    h, w = map(int, mask.shape)
    blocked = _dilate(local_occupied)
    for _ in range(tries):
        r0 = int(RNG.integers(0, n - h + 1))
        c0 = int(RNG.integers(0, n - w + 1))
        if (blocked[r0 : r0 + h, c0 : c0 + w] & mask).any():
            continue
        if (reserved_global[r0 : r0 + h, c0 : c0 + w] & mask).any():
            continue
        return r0, c0
    return None


def _special_overlap_ok(mask_tl: np.ndarray, mask_tr: np.ndarray, mask_bl: np.ndarray) -> bool:
    tl_cells = _coords_from_mask(mask_tl)
    tr_cells = _coords_from_mask(mask_tr)
    bl_cells = _coords_from_mask(mask_bl)
    if not (tl_cells & tr_cells):
        return False
    if not (tl_cells - tr_cells):
        return False
    if not (tr_cells - tl_cells):
        return False
    prior = tl_cells | tr_cells
    return bool(bl_cells & prior) and bool(bl_cells - prior)


def _build_special_masks(used_keys: set[tuple[bytes, tuple[int, int]]]) -> dict | None:
    for _ in range(200):
        shared_base, shared_key = _fresh_mask(used_keys)
        used_once = set(used_keys)
        used_once.add(shared_key)
        tl_base, tl_key = _fresh_mask(used_once)
        used_once.add(tl_key)
        bl_base, bl_key = _fresh_mask(used_once)

        tl_mask = _random_variant(tl_base)
        tr_mask = _random_variant(shared_base)
        br_mask = _random_variant(shared_base)
        bl_mask = _random_variant(bl_base)
        if _special_overlap_ok(tl_mask, tr_mask, bl_mask):
            return {
                "shared_key": shared_key,
                "tl_key": tl_key,
                "bl_key": bl_key,
                "tl_mask": tl_mask,
                "tr_mask": tr_mask,
                "bl_mask": bl_mask,
                "br_mask": br_mask,
            }
    return None


def _trace_scene(instances: list[dict], n: int) -> dict:
    shape_counts = _shape_quadrant_counts(instances)
    actual = np.zeros((n, n), dtype=int)
    opaque_overwrite = False
    transparent_block = False
    transparent_fill = False
    visible_by_object: dict[int, int] = defaultdict(int)

    for quadrant in QUADRANT_ORDER:
        qobjs = [obj for obj in instances if obj["quadrant"] == quadrant]
        qobjs.sort(key=lambda obj: (obj["r0"], obj["c0"], obj["mask"].shape[0], obj["mask"].shape[1], obj["color"]))
        for obj in qobjs:
            is_shared = bool(shape_counts[obj["shape_key"]] >= 2)
            for rr, cc in np.argwhere(obj["mask"]):
                r = int(obj["r0"] + rr)
                c = int(obj["c0"] + cc)
                before = int(actual[r, c])
                if is_shared:
                    if before != BACKGROUND and before != int(obj["color"]):
                        opaque_overwrite = True
                    actual[r, c] = int(obj["color"])
                    visible_by_object[id(obj)] += 1
                else:
                    if before == BACKGROUND:
                        actual[r, c] = int(obj["color"])
                        transparent_fill = True
                        visible_by_object[id(obj)] += 1
                    else:
                        transparent_block = True

    cross_quadrant_overlap = False
    for i, left in enumerate(instances):
        left_cells = _coords_from_mask(left["mask"], left["r0"], left["c0"])
        for right in instances[i + 1 :]:
            if left["quadrant"] == right["quadrant"]:
                continue
            right_cells = _coords_from_mask(right["mask"], right["r0"], right["c0"])
            if left_cells & right_cells:
                cross_quadrant_overlap = True
                break
        if cross_quadrant_overlap:
            break

    return {
        "shape_counts": shape_counts,
        "actual": actual,
        "all_opaque": _render_instances(instances, n, shape_counts, force="opaque"),
        "all_transparent": _render_instances(instances, n, shape_counts, force="transparent"),
        "opaque_overwrite": opaque_overwrite,
        "transparent_block": transparent_block,
        "transparent_fill": transparent_fill,
        "cross_quadrant_overlap": cross_quadrant_overlap,
        "visible_by_object": visible_by_object,
    }


def _render_input(instances: list[dict], n: int) -> np.ndarray:
    grid = np.zeros((2 * n, 2 * n), dtype=int)
    for quadrant in QUADRANT_ORDER:
        qr, qc = QUADRANT_OFFSETS[quadrant]
        for obj in [item for item in instances if item["quadrant"] == quadrant]:
            patch = np.zeros(obj["mask"].shape, dtype=int)
            patch[obj["mask"]] = int(obj["color"])
            paste(grid, patch, qr * n + int(obj["r0"]), qc * n + int(obj["c0"]), transparent=BACKGROUND)
    return grid


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    n = int(RNG.integers(QUADRANT_MIN, QUADRANT_MAX + 1))
    used_keys: set[tuple[bytes, tuple[int, int]]] = set()

    specials = _build_special_masks(used_keys)
    if specials is None:
        return None
    used_keys.update({specials["shared_key"], specials["tl_key"], specials["bl_key"]})

    max_h = max(specials["tl_mask"].shape[0], specials["tr_mask"].shape[0], specials["bl_mask"].shape[0])
    max_w = max(specials["tl_mask"].shape[1], specials["tr_mask"].shape[1], specials["bl_mask"].shape[1])
    if max_h > n or max_w > n:
        return None

    anchor_r = int(RNG.integers(0, n - max_h + 1))
    anchor_c = int(RNG.integers(0, n - max_w + 1))

    quad_objects: dict[str, list[dict]] = {quadrant: [] for quadrant in QUADRANT_ORDER}
    quadrant_occupied = {quadrant: np.zeros((n, n), dtype=bool) for quadrant in QUADRANT_ORDER}

    tl_special = {"mask": specials["tl_mask"], "r0": anchor_r, "c0": anchor_c, "shape_key": specials["tl_key"]}
    tr_special = {"mask": specials["tr_mask"], "r0": anchor_r, "c0": anchor_c, "shape_key": specials["shared_key"]}
    bl_special = {"mask": specials["bl_mask"], "r0": anchor_r, "c0": anchor_c, "shape_key": specials["bl_key"]}
    for quadrant, obj in (("TL", tl_special), ("TR", tr_special), ("BL", bl_special)):
        if not _fits(obj["mask"], n, obj["r0"], obj["c0"]):
            return None
        quad_objects[quadrant].append(obj)
        _place_on(obj["mask"], quadrant_occupied[quadrant], obj["r0"], obj["c0"])

    reserved_global = np.zeros((n, n), dtype=bool)
    for obj in (tl_special, tr_special, bl_special):
        _place_on(obj["mask"], reserved_global, obj["r0"], obj["c0"])

    br_mask = specials["br_mask"]
    br_pos = _find_position(br_mask, n, quadrant_occupied["BR"], reserved_global, tries=300)
    if br_pos is None:
        return None
    br_special = {"mask": br_mask, "r0": br_pos[0], "c0": br_pos[1], "shape_key": specials["shared_key"]}
    quad_objects["BR"].append(br_special)
    _place_on(br_special["mask"], quadrant_occupied["BR"], br_special["r0"], br_special["c0"])
    _place_on(br_special["mask"], reserved_global, br_special["r0"], br_special["c0"])

    pair_mode = str(RNG.choice(["none", "diag_a", "diag_b"], p=[0.45, 0.275, 0.275]))
    shared_pair: dict[str, tuple[np.ndarray, tuple[bytes, tuple[int, int]]]] = {}
    if pair_mode != "none":
        pair_base, pair_key = _fresh_mask(used_keys)
        used_keys.add(pair_key)
        if pair_mode == "diag_a":
            shared_pair = {"TL": (_random_variant(pair_base), pair_key), "BR": (_random_variant(pair_base), pair_key)}
        else:
            shared_pair = {"TR": (_random_variant(pair_base), pair_key), "BL": (_random_variant(pair_base), pair_key)}

    for quadrant in QUADRANT_ORDER:
        if quadrant in shared_pair:
            filler_mask, filler_key = shared_pair[quadrant]
        else:
            filler_mask, filler_key = _fresh_mask(used_keys)
            used_keys.add(filler_key)
        pos = _find_position(filler_mask, n, quadrant_occupied[quadrant], reserved_global)
        if pos is None:
            return None
        filler = {"mask": filler_mask, "r0": pos[0], "c0": pos[1], "shape_key": filler_key}
        quad_objects[quadrant].append(filler)
        _place_on(filler["mask"], quadrant_occupied[quadrant], filler["r0"], filler["c0"])
        _place_on(filler["mask"], reserved_global, filler["r0"], filler["c0"])

    for quadrant in QUADRANT_ORDER:
        if len(quad_objects[quadrant]) >= 4 or RNG.random() >= 0.35:
            continue
        extra_mask, extra_key = _fresh_mask(used_keys)
        used_keys.add(extra_key)
        pos = _find_position(extra_mask, n, quadrant_occupied[quadrant], reserved_global)
        if pos is None:
            continue
        extra = {"mask": extra_mask, "r0": pos[0], "c0": pos[1], "shape_key": extra_key}
        quad_objects[quadrant].append(extra)
        _place_on(extra["mask"], quadrant_occupied[quadrant], extra["r0"], extra["c0"])
        _place_on(extra["mask"], reserved_global, extra["r0"], extra["c0"])

    instances: list[dict] = []
    for quadrant in QUADRANT_ORDER:
        colors = RNG.choice(np.arange(1, 10), size=len(quad_objects[quadrant]), replace=False)
        for obj, color in zip(quad_objects[quadrant], colors):
            instances.append(
                {
                    "quadrant": quadrant,
                    "mask": obj["mask"],
                    "r0": int(obj["r0"]),
                    "c0": int(obj["c0"]),
                    "shape_key": obj["shape_key"],
                    "color": int(color),
                }
            )

    trace = _trace_scene(instances, n)
    shared_shapes = [key for key, count in trace["shape_counts"].items() if count >= 2]
    unique_shapes = [key for key, count in trace["shape_counts"].items() if count == 1]
    if not shared_shapes or not unique_shapes:
        return None
    if not trace["cross_quadrant_overlap"]:
        return None
    if not trace["opaque_overwrite"]:
        return None
    if not trace["transparent_block"] or not trace["transparent_fill"]:
        return None
    if np.array_equal(trace["actual"], trace["all_opaque"]):
        return None
    if np.array_equal(trace["actual"], trace["all_transparent"]):
        return None
    if not any(trace["visible_by_object"][id(obj)] > 0 for obj in instances if trace["shape_counts"][obj["shape_key"]] >= 2):
        return None
    if not any(trace["visible_by_object"][id(obj)] > 0 for obj in instances if trace["shape_counts"][obj["shape_key"]] == 1):
        return None
    if any(not (2 <= len(quad_objects[q]) <= 4) for q in QUADRANT_ORDER):
        return None

    inp = _render_input(instances, n)
    out = trace["actual"]
    if not np.array_equal(main(inp), out):
        return None

    n2, parsed = _ordered_objects(inp)
    reparsed_counts = _shape_quadrant_counts(parsed)
    if n2 != n or reparsed_counts != trace["shape_counts"]:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_scene()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
