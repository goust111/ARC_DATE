import importlib.util
import random
from collections import Counter, deque
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")

try:
    _spec = importlib.util.spec_from_file_location("_shared_arc_utils_010_5", _ARC_UTILS_PATH)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"cannot load {_ARC_UTILS_PATH}")
    _arc_utils = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_arc_utils)
    Color = _arc_utils.Color
    find_components = _arc_utils.find_components
    get_bg = _arc_utils.get_bg
except Exception:
    class Color:
        BLACK = 0
        RED = 2

    def get_bg(grid: np.ndarray) -> int:
        values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
        return int(values[np.argmax(counts)])

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors: list[int] | None = None,
        min_area: int = 1,
        return_format: str = "dict",
    ) -> list[dict]:
        if not monochromatic:
            raise NotImplementedError("fallback only supports monochromatic=True")
        if return_format != "dict":
            raise NotImplementedError("fallback only supports return_format='dict'")
        g = np.asarray(grid, dtype=int)
        if bg is None:
            bg = get_bg(g)
        bg = int(bg)
        if colors is None:
            allowed = set(map(int, np.unique(g)))
            allowed.discard(bg)
        else:
            allowed = {int(c) for c in colors if int(c) != bg}
        if connectivity == 4:
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

        seen = np.zeros(g.shape, dtype=bool)
        out: list[dict] = []
        height, width = g.shape
        for r in range(height):
            for c in range(width):
                color = int(g[r, c])
                if seen[r, c] or color == bg or color not in allowed:
                    continue
                q = deque([(r, c)])
                seen[r, c] = True
                coords = []
                while q:
                    rr, cc = q.popleft()
                    coords.append((rr, cc))
                    for dr, dc in steps:
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(g[nr, nc]) == color:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                if len(coords) < int(min_area):
                    continue
                arr = np.asarray(coords, dtype=int)
                r0, c0 = arr.min(axis=0)
                r1, c1 = arr.max(axis=0)
                out.append(
                    {
                        "color": color,
                        "coords": arr,
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "area": int(len(coords)),
                    }
                )
        return out


BG = Color.BLACK
SHADOW = Color.RED
BODY_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
CONNECTIVITY = 4
MAX_GENERATION_ATTEMPTS = 400
GRID_SIZE_RANGE = (20, 29)
OBJECT_COUNT_RANGE = (3, 5)
OFFSET_MAG_RANGE = (5, 8)
MASK_SIDE_RANGE = range(2, 5)
MIN_MASK_AREA = 3
PLACEMENT_GAP = 1


def _mask_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    q = deque([tuple(map(int, cells[0]))])
    seen = {q[0]}
    height, width = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _enumerate_masks() -> tuple[list[np.ndarray], list[np.ndarray]]:
    symmetric: list[np.ndarray] = []
    asymmetric: list[np.ndarray] = []
    for height in MASK_SIDE_RANGE:
        for width in MASK_SIDE_RANGE:
            area = height * width
            for bits in range(1, 1 << area):
                bit_count = bits.bit_count()
                if bit_count < MIN_MASK_AREA:
                    continue
                mask = np.zeros((height, width), dtype=bool)
                for idx in range(area):
                    if bits & (1 << idx):
                        mask[idx // width, idx % width] = True
                if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                    continue
                if not _mask_connected(mask):
                    continue
                if np.array_equal(mask, mask[::-1, ::-1]):
                    symmetric.append(mask)
                else:
                    asymmetric.append(mask)
    return symmetric, asymmetric


SYMMETRIC_MASKS, ASYMMETRIC_MASKS = _enumerate_masks()


def _mask_from_component(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, component["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = np.asarray(component["coords"], dtype=int)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _shift_bbox(bbox: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    return r0 + int(dr), c0 + int(dc), r1 + int(dr), c1 + int(dc)


def _bbox_shape(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    return r1 - r0 + 1, c1 - c0 + 1


def _bbox_in_bounds(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    height, width = map(int, shape)
    r0, c0, r1, c1 = map(int, bbox)
    return 0 <= r0 <= r1 < height and 0 <= c0 <= c1 < width


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _expand_rect(rect: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = map(int, shape)
    r0, c0, r1, c1 = map(int, rect)
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(height - 1, r1 + pad),
        min(width - 1, c1 + pad),
    )


def _absolute_coords(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    return [(int(r) + int(top), int(c) + int(left)) for r, c in coords]


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    coords = np.argwhere(mask)
    for r, c in coords:
        grid[int(top) + int(r), int(left) + int(c)] = int(color)


def _is_half_turn_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, mask[::-1, ::-1]))


def _candidate_vectors(bodies: list[dict], shadows: list[dict]) -> set[tuple[int, int]]:
    candidates: set[tuple[int, int]] = set()
    for body in bodies:
        if int(body["area"]) <= 1:
            continue
        body_mask = _mask_from_component(body)
        target_mask = body_mask[::-1, ::-1]
        br0, bc0, _br1, _bc1 = map(int, body["bbox"])
        shape = body_mask.shape
        for shadow in shadows:
            if _bbox_shape(body["bbox"]) != _bbox_shape(shadow["bbox"]):
                continue
            if not np.array_equal(_mask_from_component(shadow), target_mask):
                continue
            sr0, sc0, _sr1, _sc1 = map(int, shadow["bbox"])
            candidates.add((sr0 - br0, sc0 - bc0))
    return candidates


def _analyze_vector(
    grid: np.ndarray,
    bodies: list[dict],
    shadows: list[dict],
    vector: tuple[int, int],
) -> dict | None:
    dr, dc = map(int, vector)
    if dr == 0 and dc == 0:
        return None

    shadow_by_bbox = {tuple(map(int, comp["bbox"])): comp for comp in shadows}
    used_shadow_boxes: set[tuple[int, int, int, int]] = set()
    symmetric_pairs = 0
    asymmetric_pairs = 0
    singletons = [comp for comp in bodies if int(comp["area"]) == 1]

    for body in bodies:
        if int(body["area"]) == 1:
            continue
        target_bbox = _shift_bbox(body["bbox"], dr, dc)
        shadow = shadow_by_bbox.get(target_bbox)
        if shadow is None or target_bbox in used_shadow_boxes:
            return None
        if not np.array_equal(_mask_from_component(shadow), _mask_from_component(body)[::-1, ::-1]):
            return None
        used_shadow_boxes.add(target_bbox)
        if _is_half_turn_symmetric(_mask_from_component(body)):
            symmetric_pairs += 1
        else:
            asymmetric_pairs += 1

    if symmetric_pairs < 1 or asymmetric_pairs < 1:
        return None
    if len(singletons) != 1:
        return None

    unmatched_shadows = [comp for comp in shadows if tuple(map(int, comp["bbox"])) not in used_shadow_boxes]
    if len(unmatched_shadows) != 1:
        return None

    relic = singletons[0]
    relic_coord = tuple(map(int, np.asarray(relic["coords"], dtype=int)[0]))
    if int(relic["color"]) in (BG, SHADOW):
        return None

    shadow = unmatched_shadows[0]
    shadow_mask = _mask_from_component(shadow)
    if _is_half_turn_symmetric(shadow_mask):
        return None
    missing_bbox = _shift_bbox(shadow["bbox"], -dr, -dc)
    if not _bbox_in_bounds(grid.shape, missing_bbox):
        return None
    missing_mask = shadow_mask[::-1, ::-1]
    body_coords = _absolute_coords(missing_mask, missing_bbox[0], missing_bbox[1])
    body_coord_set = set(body_coords)
    if relic_coord not in body_coord_set:
        return None

    visible = {(int(r), int(c)) for r, c in body_coords if int(grid[r, c]) != BG}
    if visible != {relic_coord}:
        return None

    return {
        "vector": (dr, dc),
        "relic_color": int(relic["color"]),
        "missing_coords": sorted(body_coord_set),
    }


def _solve_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg == BG

    components = find_components(
        g,
        bg=bg,
        connectivity=CONNECTIVITY,
        monochromatic=True,
        return_format="dict",
    )
    shadows = [comp for comp in components if int(comp["color"]) == SHADOW]
    bodies = [comp for comp in components if int(comp["color"]) not in (bg, SHADOW)]
    assert shadows and bodies

    candidates = _candidate_vectors(bodies, shadows)
    analyses = []
    for vector in sorted(candidates):
        info = _analyze_vector(g, bodies, shadows, vector)
        if info is not None:
            analyses.append(info)
    assert len(analyses) == 1
    return analyses[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the erased body by reversing the shared rotate-180-then-translate shadow rule.

    Input structure:
    - The grid has a black background, several monochromatic body objects, and one red shadow for each object.
    - Every red shadow is the 180-degree rotation of its body inside the same bbox shape, then shifted by one global offset; shadows and bodies do not overlap.
    - Exactly one body's shadow remains complete while the body itself has been erased down to a single relic pixel of its original color.

    Rule:
    - Use the intact body-shadow pairs to infer the common shadow offset and confirm that the shadow copies are rotated by 180 degrees before that translation.
    - Find the lone red shadow without a full body partner, shift it back by the common offset, rotate it by 180 degrees, and recover the missing body's footprint.
    - Fill that recovered footprint with the relic pixel's color while leaving every other visible pixel unchanged.

    Key insight:
    - The red shape is evidence, not damage: an asymmetric intact pair reveals that the true relation is rotate-180 then translate, which lets the isolated shadow be inverted uniquely.
    """
    g = np.asarray(grid, dtype=int)
    info = _solve_instance(g)
    out = g.copy()
    for r, c in info["missing_coords"]:
        out[int(r), int(c)] = int(info["relic_color"])
    return out


def _random_sign() -> int:
    return random.choice([-1, 1])


def _random_offset() -> tuple[int, int]:
    return (
        _random_sign() * random.randint(*OFFSET_MAG_RANGE),
        _random_sign() * random.randint(*OFFSET_MAG_RANGE),
    )


def _sample_masks(n_objects: int) -> list[tuple[np.ndarray, bool, bool]]:
    assert n_objects >= 3
    specs = [
        (random.choice(ASYMMETRIC_MASKS), False, False),
        (random.choice(SYMMETRIC_MASKS), True, False),
        (random.choice(ASYMMETRIC_MASKS), False, True),
    ]
    for _ in range(n_objects - 3):
        is_symmetric = random.choice([True, False])
        bank = SYMMETRIC_MASKS if is_symmetric else ASYMMETRIC_MASKS
        specs.append((random.choice(bank), is_symmetric, False))
    random.shuffle(specs)
    return specs


def _valid_body_starts(
    grid_shape: tuple[int, int],
    mask_shape: tuple[int, int],
    vector: tuple[int, int],
) -> list[tuple[int, int]]:
    height, width = map(int, grid_shape)
    mh, mw = map(int, mask_shape)
    dr, dc = map(int, vector)
    row_lo = max(0, -dr)
    row_hi = height - mh - max(0, dr)
    col_lo = max(0, -dc)
    col_hi = width - mw - max(0, dc)
    if row_lo > row_hi or col_lo > col_hi:
        return []
    cells = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
    random.shuffle(cells)
    return cells


def _place_objects(
    grid_shape: tuple[int, int],
    vector: tuple[int, int],
    specs: list[tuple[np.ndarray, bool, bool]],
) -> list[dict] | None:
    ordered = sorted(
        specs,
        key=lambda item: (-int(item[0].shape[0] * item[0].shape[1]), -int(item[0].sum())),
    )
    placed: list[dict] = []
    forbidden: list[tuple[int, int, int, int]] = []

    for mask, is_symmetric, is_missing in ordered:
        starts = _valid_body_starts(grid_shape, mask.shape, vector)
        if not starts:
            return None
        chosen = None
        for top, left in starts:
            body_bbox = (top, left, top + mask.shape[0] - 1, left + mask.shape[1] - 1)
            shadow_bbox = _shift_bbox(body_bbox, vector[0], vector[1])
            if any(_rect_overlap(body_bbox, rect) or _rect_overlap(shadow_bbox, rect) for rect in forbidden):
                continue
            chosen = {
                "mask": mask,
                "symmetric": is_symmetric,
                "missing": is_missing,
                "color": None,
                "body_top": (top, left),
                "shadow_top": (top + vector[0], left + vector[1]),
                "body_bbox": body_bbox,
                "shadow_bbox": shadow_bbox,
            }
            forbidden.append(_expand_rect(body_bbox, PLACEMENT_GAP, grid_shape))
            forbidden.append(_expand_rect(shadow_bbox, PLACEMENT_GAP, grid_shape))
            placed.append(chosen)
            break
        if chosen is None:
            return None
    return placed


def _render_scene(
    grid_shape: tuple[int, int],
    objects: list[dict],
    body_colors: list[int],
) -> tuple[np.ndarray, np.ndarray, dict]:
    inp = np.full(grid_shape, BG, dtype=int)
    out = np.full(grid_shape, BG, dtype=int)
    missing_spec = None
    for obj, color in zip(objects, body_colors):
        obj["color"] = int(color)
        mask = obj["mask"]
        body_top = obj["body_top"]
        shadow_top = obj["shadow_top"]
        shadow_mask = mask[::-1, ::-1]
        _paint_mask(out, mask, body_top[0], body_top[1], color)
        _paint_mask(out, shadow_mask, shadow_top[0], shadow_top[1], SHADOW)
        _paint_mask(inp, shadow_mask, shadow_top[0], shadow_top[1], SHADOW)
        if obj["missing"]:
            coords = _absolute_coords(mask, body_top[0], body_top[1])
            relic = random.choice(coords)
            inp[relic[0], relic[1]] = color
            missing_spec = {
                "color": color,
                "coords": set(coords),
                "relic": relic,
                "shadow_coords": set(_absolute_coords(shadow_mask, shadow_top[0], shadow_top[1])),
            }
        else:
            _paint_mask(inp, mask, body_top[0], body_top[1], color)
    assert missing_spec is not None
    return inp, out, missing_spec


def _semantic_check(
    inp: np.ndarray,
    out: np.ndarray,
    vector: tuple[int, int],
    objects: list[dict],
    missing_spec: dict,
) -> bool:
    try:
        info = _solve_instance(inp)
    except AssertionError:
        return False

    if tuple(info["vector"]) != tuple(map(int, vector)):
        return False
    if int(info["relic_color"]) != int(missing_spec["color"]):
        return False
    if set(info["missing_coords"]) != set(missing_spec["coords"]):
        return False
    if not np.array_equal(main(inp), out):
        return False

    if sum(1 for obj in objects if obj["symmetric"] and not obj["missing"]) < 1:
        return False
    if sum(1 for obj in objects if (not obj["symmetric"]) and (not obj["missing"])) < 1:
        return False
    missing_objects = [obj for obj in objects if obj["missing"]]
    if len(missing_objects) != 1:
        return False
    if _is_half_turn_symmetric(missing_objects[0]["mask"]):
        return False
    if np.count_nonzero(inp != out) != int(len(missing_spec["coords"]) - 1):
        return False

    wrong_translation_coords = set(_absolute_coords(missing_objects[0]["mask"][::-1, ::-1], missing_objects[0]["body_top"][0], missing_objects[0]["body_top"][1]))
    if wrong_translation_coords == set(missing_spec["coords"]):
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    grid_shape = (
        random.randint(*GRID_SIZE_RANGE),
        random.randint(*GRID_SIZE_RANGE),
    )
    vector = _random_offset()
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    specs = _sample_masks(n_objects)
    objects = _place_objects(grid_shape, vector, specs)
    if objects is None:
        return None
    colors = random.sample(BODY_COLORS, len(objects))
    inp, out, missing_spec = _render_scene(grid_shape, objects, colors)
    if not _semantic_check(inp, out, vector, objects, missing_spec):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shadow-reversal ARC instance")
