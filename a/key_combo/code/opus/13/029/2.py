import random
from collections import Counter, deque

import numpy as np


BG = 0
RED = 2
GREEN = 3
QUADRANTS = ("tl", "tr", "bl", "br")
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
RNG = np.random.default_rng()
MAX_GEN_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    return bbox_coords(coords)


def _crop_bbox(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(arr)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
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
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _canonical_shape_key(mask: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    best: tuple[bytes, tuple[int, int]] | None = None
    for k in range(4):
        rot = np.rot90(mask, k)
        for variant in (rot, np.fliplr(rot)):
            key = (variant.tobytes(), tuple(map(int, variant.shape)))
            if best is None or key < best:
                best = key
    assert best is not None
    return best


def _extract_object_info(grid: np.ndarray, coords: np.ndarray) -> dict:
    comp = np.asarray(coords, dtype=int)
    green = comp[grid[comp[:, 0], comp[:, 1]] == GREEN]
    red = comp[grid[comp[:, 0], comp[:, 1]] == RED]
    assert green.shape[0] == 4, "each component must contain exactly one 2x2 green marker"
    assert red.shape[0] >= 1, "each component must contain red pixels"

    mr0, mc0, mr1, mc1 = bbox_coords(green)
    assert (mr1 - mr0 + 1, mc1 - mc0 + 1) == (2, 2)
    assert np.all(grid[mr0 : mr1 + 1, mc0 : mc1 + 1] == GREEN)

    completed = set()
    row_sum = mr0 + mr1
    col_sum = mc0 + mc1
    for r, c in red:
        r = int(r)
        c = int(c)
        for rr in (r, row_sum - r):
            for cc in (c, col_sum - c):
                completed.add((int(rr), int(cc)))

    red_coords = np.array(sorted(completed), dtype=int)
    all_coords = np.vstack(
        [
            red_coords,
            np.array(
                [(mr0, mc0), (mr0, mc1), (mr1, mc0), (mr1, mc1)],
                dtype=int,
            ),
        ]
    )
    r0, c0, r1, c1 = bbox_coords(all_coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    patch[red_coords[:, 0] - r0, red_coords[:, 1] - c0] = RED
    patch[mr0 - r0 : mr1 - r0 + 1, mc0 - c0 : mc1 - c0 + 1] = GREEN

    red_mask = np.zeros(patch.shape, dtype=bool)
    red_mask[red_coords[:, 0] - r0, red_coords[:, 1] - c0] = True
    return {
        "bbox": (r0, c0, r1, c1),
        "patch": patch,
        "shape_key": _canonical_shape_key(red_mask),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each marker-centered fragment by four-way reflection, then sort objects to the top or bottom by the completed shape's frequency.

    Input structure:
    - A black background contains 4-6 scattered 8-connected multicolor components.
    - Each component has a solid green 2x2 marker square and several red pixels in one or two quadrants around it.
    - Components are separated so each marker belongs to exactly one local fragment.

    Rule:
    - For each component, use the green 2x2 block as the reflection center and mirror every red pixel horizontally, vertically, and by 180 degrees to recover the full red shape.
    - Canonicalize the completed red mask up to the 8 rotation/flip variants and count how often each canonical shape appears.
    - Move every completed object whose canonical shape appears at least twice to the top edge, and move every singleton shape object to the bottom edge.
    - Preserve each completed object's horizontal position and keep both the reflected red pixels and the green marker square.

    Key insight:
    - The grouping signal is hidden in the completed red shapes, not in the visible fragments: different-looking partial pieces can become the same shape after reflection.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    assert bg == BG

    comps = _find_components(g != bg)
    assert comps, "expected at least one foreground component"
    objects = [_extract_object_info(g, comp) for comp in comps]
    freq = Counter(obj["shape_key"] for obj in objects)

    out = np.full((h, w), bg, dtype=int)
    for obj in objects:
        r0, c0, r1, _c1 = obj["bbox"]
        patch = obj["patch"]
        ph = patch.shape[0]
        dest_r0 = 0 if freq[obj["shape_key"]] >= 2 else h - ph
        region = out[dest_r0 : dest_r0 + ph, c0 : c0 + patch.shape[1]]
        assert np.all((region == bg) | (patch == bg))
        region[patch != bg] = patch[patch != bg]
    return out


def _orbit_cell(a: int, b: int, dr: int, dc: int, quadrant: str) -> tuple[int, int]:
    if quadrant == "tl":
        return a - dr, b - dc
    if quadrant == "tr":
        return a - dr, b + 1 + dc
    if quadrant == "bl":
        return a + 1 + dr, b - dc
    assert quadrant == "br"
    return a + 1 + dr, b + 1 + dc


def _build_red_mask(a: int, b: int, orbits: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((2 * a + 2, 2 * b + 2), dtype=bool)
    for dr, dc in orbits:
        for quadrant in QUADRANTS:
            rr, cc = _orbit_cell(a, b, dr, dc, quadrant)
            mask[rr, cc] = True
    return mask


def _build_patch(a: int, b: int, orbits: set[tuple[int, int]], quadrants: tuple[str, ...]) -> np.ndarray:
    patch = np.full((2 * a + 2, 2 * b + 2), BG, dtype=int)
    patch[a : a + 2, b : b + 2] = GREEN
    for dr, dc in orbits:
        for quadrant in quadrants:
            rr, cc = _orbit_cell(a, b, dr, dc, quadrant)
            patch[rr, cc] = RED
    return patch


def _connected_orbits(a: int, b: int) -> set[tuple[int, int]]:
    cells = [(dr, dc) for dr in range(1, a + 1) for dc in range(1, b + 1)]
    max_size = len(cells)
    if max_size == 2:
        target = 2
    else:
        target = int(RNG.integers(2, max_size + 1))
    chosen = {(1, 1)}
    while len(chosen) < target:
        frontier = []
        for dr, dc in chosen:
            for ndr in range(max(1, dr - 1), min(a, dr + 1) + 1):
                for ndc in range(max(1, dc - 1), min(b, dc + 1) + 1):
                    cell = (ndr, ndc)
                    if cell not in chosen:
                        frontier.append(cell)
        assert frontier
        chosen.add(frontier[int(RNG.integers(0, len(frontier)))])
    return chosen


def _make_template() -> dict:
    dims = [(1, 2), (2, 1), (2, 2)]
    a, b = dims[int(RNG.integers(0, len(dims)))]
    orbits = _connected_orbits(a, b)
    raw_full_patch = _build_patch(a, b, orbits, QUADRANTS)
    full_bbox = _tight_bbox(raw_full_patch != BG)
    full_patch = _crop_bbox(raw_full_patch, full_bbox)
    red_mask = full_patch == RED
    return {
        "a": a,
        "b": b,
        "orbits": orbits,
        "full_bbox": full_bbox,
        "full_patch": full_patch,
        "red_mask": red_mask,
        "shape_key": _canonical_shape_key(red_mask),
        "h": int(red_mask.shape[0]),
        "w": int(red_mask.shape[1]),
    }


def _choose_partial_plans(orbits: set[tuple[int, int]], count: int) -> list[tuple[str, ...]]:
    orbit_count = len(orbits)
    singles = [(q,) for q in QUADRANTS if orbit_count >= 3]
    pairs = [tuple(pair) for pair in (
        ("tl", "tr"), ("tl", "bl"), ("tl", "br"),
        ("tr", "bl"), ("tr", "br"), ("bl", "br"),
    )]
    plans = singles + pairs
    RNG.shuffle(plans)
    out: list[tuple[str, ...]] = []
    used: set[tuple[str, ...]] = set()
    for plan in plans:
        visible = orbit_count * len(plan)
        if 3 <= visible <= 8 and plan not in used:
            out.append(plan)
            used.add(plan)
            if len(out) == count:
                return out
    while len(out) < count:
        plan = plans[int(RNG.integers(0, len(plans)))]
        visible = orbit_count * len(plan)
        if 3 <= visible <= 8:
            out.append(plan)
    return out


def _duplicate_group_layout(n_objects: int) -> list[int]:
    options = {
        4: [[2]],
        5: [[2], [3], [2, 2]],
        6: [[3], [2, 2], [3, 2]],
    }
    choices = options[n_objects]
    return list(choices[int(RNG.integers(0, len(choices)))])


def _build_object_specs() -> list[dict] | None:
    n_objects = int(RNG.integers(4, 7))
    dup_sizes = _duplicate_group_layout(n_objects)
    unique_count = n_objects - sum(dup_sizes)
    if unique_count < 1:
        return None

    templates: list[dict] = []
    used_keys: set[tuple[bytes, tuple[int, int]]] = set()
    group_defs = [{"size": size, "shared": True} for size in dup_sizes]
    group_defs.extend({"size": 1, "shared": False} for _ in range(unique_count))

    for group in group_defs:
        for _ in range(100):
            template = _make_template()
            if template["shape_key"] in used_keys:
                continue
            plans = _choose_partial_plans(template["orbits"], group["size"])
            if len(plans) < group["size"]:
                continue
            template = dict(template)
            template["plans"] = plans
            templates.append(template)
            used_keys.add(template["shape_key"])
            break
        else:
            return None

    objects = []
    for template, group in zip(templates, group_defs):
        for idx in range(group["size"]):
            objects.append(
                {
                    "shared": group["shared"],
                    "shape_key": template["shape_key"],
                    "h": template["h"],
                    "w": template["w"],
                    "full_patch": template["full_patch"],
                    "input_patch": _crop_bbox(
                        _build_patch(
                            template["a"],
                            template["b"],
                            template["orbits"],
                            template["plans"][idx],
                        ),
                        template["full_bbox"],
                    ),
                }
            )

    RNG.shuffle(objects)
    shared = [obj for obj in objects if obj["shared"]]
    if len(shared) < 2:
        return None
    return objects


def _assign_columns(widths: list[int], width: int) -> list[int] | None:
    gap = 1
    required = 4 + sum(widths) + gap * max(0, len(widths) - 1)
    if required > width:
        return None
    slack = width - required
    extras = [0] * (len(widths) + 1)
    for _ in range(slack):
        extras[int(RNG.integers(0, len(extras)))] += 1
    c0 = 2 + extras[0]
    cols = []
    for idx, obj_w in enumerate(widths):
        cols.append(c0)
        c0 += obj_w + gap + extras[idx + 1]
    return cols


def _choose_canvas(objects: list[dict]) -> tuple[int, int] | None:
    top_widths = [obj["w"] for obj in objects if obj["shared"]]
    bottom_widths = [obj["w"] for obj in objects if not obj["shared"]]
    if not top_widths or not bottom_widths:
        return None
    req_w = max(
        20,
        4 + sum(top_widths) + max(0, len(top_widths) - 1),
        4 + sum(bottom_widths) + max(0, len(bottom_widths) - 1),
    )
    max_top_h = max(obj["h"] for obj in objects if obj["shared"])
    max_bottom_h = max(obj["h"] for obj in objects if not obj["shared"])
    req_h = max(20, max_top_h + max_bottom_h + 8)
    if req_w > 30 or req_h > 30:
        return None
    h = int(RNG.integers(req_h, 31))
    w = int(RNG.integers(req_w, 31))
    return h, w


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _scatter_rows(objects: list[dict], height: int) -> list[int] | None:
    order = list(range(len(objects)))
    RNG.shuffle(order)
    rows = [None] * len(objects)
    placed_rects: list[tuple[int, int, int, int]] = []
    for idx in order:
        obj = objects[idx]
        obj_h = obj["h"]
        choices = list(range(2, height - obj_h - 1))
        if not choices:
            return None
        RNG.shuffle(choices)
        for r0 in choices:
            rect = (r0, obj["c0"], r0 + obj_h - 1, obj["c0"] + obj["w"] - 1)
            if any(_rects_overlap(rect, other, margin=1) for other in placed_rects):
                continue
            rows[idx] = r0
            placed_rects.append(rect)
            break
        else:
            return None
    return [int(r) for r in rows]


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    region = grid[r0 : r0 + h, c0 : c0 + w]
    assert np.all((region == BG) | (patch == BG))
    region[patch != BG] = patch[patch != BG]


def _gen_attempt() -> np.ndarray | None:
    objects = _build_object_specs()
    if objects is None:
        return None
    canvas = _choose_canvas(objects)
    if canvas is None:
        return None
    h, w = canvas

    top_indices = [i for i, obj in enumerate(objects) if obj["shared"]]
    bottom_indices = [i for i, obj in enumerate(objects) if not obj["shared"]]
    top_cols = _assign_columns([objects[i]["w"] for i in top_indices], w)
    bottom_cols = _assign_columns([objects[i]["w"] for i in bottom_indices], w)
    if top_cols is None or bottom_cols is None:
        return None

    for idx, c0 in zip(top_indices, top_cols):
        objects[idx]["c0"] = int(c0)
    for idx, c0 in zip(bottom_indices, bottom_cols):
        objects[idx]["c0"] = int(c0)

    rows = _scatter_rows(objects, h)
    if rows is None:
        return None
    for obj, r0 in zip(objects, rows):
        obj["input_r0"] = int(r0)

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    for obj in objects:
        _place_patch(inp, obj["input_patch"], obj["input_r0"], obj["c0"])
        out_r0 = 0 if obj["shared"] else h - obj["h"]
        _place_patch(out, obj["full_patch"], out_r0, obj["c0"])

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for opus/13/029/2")
