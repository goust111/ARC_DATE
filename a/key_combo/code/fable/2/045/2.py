import random

import numpy as np


class Color:
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

    OBJECT_POOL = (BLUE, RED, GREEN, GREY, PINK, ORANGE, TEAL, MAROON)


TRANSFORMS = ("flip_ud", "flip_lr", "diag_main", "diag_anti")
SLOT_FAMILIES = {
    "flip_ud": ("top", "bottom"),
    "flip_lr": ("left", "right"),
    "diag_main": ("tl", "br"),
    "diag_anti": ("tr", "bl"),
}
SLOT_TO_OP = {slot: op for op, slots in SLOT_FAMILIES.items() for slot in slots}

CANVAS_SIZE = 30
PLACEMENT_BORDER = 2
PLACEMENT_GAP = 1
OBJECT_COUNT_RANGE = (4, 5)
SIZE_CHOICES = (5, 7)
GENERATION_RETRIES = 400
SHAPE_RETRIES = 120
PLACEMENT_RETRIES = 250


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_bool(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        coords = []
        while stack:
            cr, cc2 = stack.pop()
            coords.append((cr, cc2))
            for nr, nc in _neighbors4(cr, cc2):
                if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        parts.append(np.array(coords, dtype=int))
    return parts


def find_components(
    grid: np.ndarray,
    *,
    bg: int = Color.BLACK,
    colors: list[int] | tuple[int, ...] | None = None,
) -> list[dict]:
    if colors is None:
        colors = sorted(set(map(int, np.unique(grid))) - {int(bg)})
    out = []
    for color in colors:
        col = int(color)
        if col == int(bg):
            continue
        for coords in _components_bool(grid == col):
            out.append(
                {
                    "color": col,
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _special_slots(size: int) -> dict[str, tuple[int, int]]:
    mid = size // 2
    last = size - 1
    return {
        "top": (0, mid),
        "bottom": (last, mid),
        "left": (mid, 0),
        "right": (mid, last),
        "tl": (0, 0),
        "tr": (0, last),
        "bl": (last, 0),
        "br": (last, last),
    }


def _slot_name(size: int, coord: tuple[int, int]) -> str | None:
    for name, pos in _special_slots(size).items():
        if pos == coord:
            return name
    return None


def _apply_transform(mask: np.ndarray, op: str) -> np.ndarray:
    if op == "flip_ud":
        return np.flipud(mask).copy()
    if op == "flip_lr":
        return np.fliplr(mask).copy()
    if op == "diag_main":
        return mask.T.copy()
    if op == "diag_anti":
        return np.fliplr(np.flipud(mask)).T.copy()
    raise ValueError(f"unknown transform: {op}")


def _array_key(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, arr.shape)), arr.astype(np.uint8).tobytes()


def _parse_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    yellow_coords = {tuple(map(int, rc)) for rc in np.argwhere(g == Color.YELLOW)}
    object_colors = sorted(set(map(int, np.unique(g))) - {bg, Color.YELLOW})
    components = find_components(g, bg=bg, colors=object_colors)
    objects = []
    used_markers = set()

    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        size = int(r1 - r0 + 1)
        assert size == int(c1 - c0 + 1), "object bbox must be square"
        assert size % 2 == 1, "object bbox side length must be odd"

        patch = crop_bbox(g, comp["bbox"])
        markers = np.argwhere(patch == Color.YELLOW)
        assert markers.shape[0] == 1, "each object bbox must contain one yellow marker"
        mr, mc = map(int, markers[0])
        marker_name = _slot_name(size, (mr, mc))
        assert marker_name is not None, "marker must lie at a corner or side midpoint"

        mask = np.zeros((size, size), dtype=bool)
        local_coords = comp["coords"] - np.array([[r0, c0]], dtype=int)
        mask[local_coords[:, 0], local_coords[:, 1]] = True
        assert not mask[mr, mc], "marker cell must be outside the object"
        assert mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()

        abs_marker = (r0 + mr, c0 + mc)
        used_markers.add(abs_marker)
        objects.append(
            {
                "color": int(comp["color"]),
                "bbox": (r0, c0, r1, c1),
                "mask": mask,
                "marker": (mr, mc),
                "marker_name": marker_name,
                "op": SLOT_TO_OP[marker_name],
            }
        )

    assert objects, "expected at least one foreground object"
    assert used_markers == yellow_coords, "all yellow pixels must be assigned as markers"
    return sorted(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["color"]))


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror each object inside its odd square bbox according to the yellow marker position.

    Input structure:
    - A black background contains several separated monochromatic foreground objects.
    - Each object's tight bbox is an odd-sized square.
    - Inside each bbox there is exactly one yellow marker pixel at a corner cell or a side-midpoint cell, and that cell is not part of the object.

    Rule:
    - If the marker is at the midpoint of the top or bottom side, flip the object vertically inside its bbox.
    - If the marker is at the midpoint of the left or right side, flip the object horizontally inside its bbox.
    - If the marker is at the top-left or bottom-right corner, reflect the object across the main diagonal of the bbox.
    - If the marker is at the top-right or bottom-left corner, reflect the object across the anti-diagonal of the bbox, then remove the marker.

    Key insight:
    - The yellow pixel is an opcode: edge-midpoint markers choose an axial mirror family, corner markers choose a diagonal family, and the marker's local bbox position selects the exact mirror axis.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for obj in _parse_objects(g):
        r0, c0, r1, c1 = obj["bbox"]
        size = r1 - r0 + 1
        mirrored = _apply_transform(obj["mask"], obj["op"])
        patch = np.full((size, size), Color.BLACK, dtype=int)
        patch[mirrored] = int(obj["color"])
        out[r0 : r1 + 1, c0 : c1 + 1] = patch

    return out


def _boundary_cells(size: int, side: str) -> list[tuple[int, int]]:
    last = size - 1
    if side == "top":
        return [(0, c) for c in range(size)]
    if side == "bottom":
        return [(last, c) for c in range(size)]
    if side == "left":
        return [(r, 0) for r in range(size)]
    if side == "right":
        return [(r, last) for r in range(size)]
    raise ValueError(f"unknown side: {side}")


def _random_progress_path(
    size: int, start: tuple[int, int], end: tuple[int, int], blocked: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    path = [start]
    r, c = start
    while (r, c) != end:
        dist = abs(r - end[0]) + abs(c - end[1])
        candidates = []
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds((size, size), nr, nc) or (nr, nc) in blocked:
                continue
            if abs(nr - end[0]) + abs(nc - end[1]) < dist:
                candidates.append((nr, nc))
        assert candidates
        r, c = random.choice(candidates)
        path.append((r, c))
    return path


def _frontier(mask: np.ndarray, blocked: set[tuple[int, int]]) -> list[tuple[int, int]]:
    size = mask.shape[0]
    seen = set()
    cells = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if not _in_bounds((size, size), nr, nc):
                continue
            if mask[nr, nc] or (nr, nc) in blocked or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            cells.append((nr, nc))
    return cells


def _mask_is_single_component(mask: np.ndarray) -> bool:
    parts = _components_bool(mask)
    return len(parts) == 1


def _mask_variants_unique(mask: np.ndarray) -> bool:
    keys = {_array_key(_apply_transform(mask, op)) for op in TRANSFORMS}
    return len(keys) == len(TRANSFORMS)


def _valid_mask(mask: np.ndarray, marker_name: str) -> bool:
    size = mask.shape[0]
    marker = _special_slots(size)[marker_name]
    op = SLOT_TO_OP[marker_name]
    area = int(mask.sum())
    min_area = max(size + 2, (size * size) // 3)
    max_area = (size * size * 2) // 3

    if mask[marker]:
        return False
    if area < min_area or area > max_area:
        return False
    if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
        return False
    if not _mask_is_single_component(mask):
        return False
    mirrored = _apply_transform(mask, op)
    if np.array_equal(mask, mirrored):
        return False
    if not _mask_variants_unique(mask):
        return False
    return True


def _build_random_mask(size: int, marker_name: str) -> np.ndarray:
    marker = _special_slots(size)[marker_name]
    blocked = {marker}
    center = (size // 2, size // 2)
    anchors = {
        side: random.choice([cell for cell in _boundary_cells(size, side) if cell not in blocked])
        for side in ("top", "bottom", "left", "right")
    }

    mask = np.zeros((size, size), dtype=bool)
    mask[center] = True
    for anchor in anchors.values():
        for cell in _random_progress_path(size, center, anchor, blocked):
            mask[cell] = True

    lower = max(int(mask.sum()), (size * size) // 3)
    upper = max(lower, (size * size * 3) // 5)
    target = random.randint(lower, upper)
    while int(mask.sum()) < target:
        options = _frontier(mask, blocked)
        if not options:
            break
        cell = random.choice(options)
        mask[cell] = True
        if random.random() < 0.25 and int(mask.sum()) < target:
            extras = [nbr for nbr in _frontier(mask, blocked) if abs(nbr[0] - cell[0]) + abs(nbr[1] - cell[1]) == 1]
            if extras:
                mask[random.choice(extras)] = True

    return mask


def _make_object_mask(size: int, marker_name: str) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        mask = _build_random_mask(size, marker_name)
        if _valid_mask(mask, marker_name):
            return mask
    raise RuntimeError("failed to build a non-degenerate object mask")


def _rect_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = PLACEMENT_GAP
) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_specs(specs: list[dict]) -> list[tuple[int, int]] | None:
    placements = [None] * len(specs)
    boxes = []
    order = sorted(range(len(specs)), key=lambda idx: specs[idx]["size"], reverse=True)
    for idx in order:
        size = specs[idx]["size"]
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(PLACEMENT_BORDER, CANVAS_SIZE - PLACEMENT_BORDER - size)
            c = random.randint(PLACEMENT_BORDER, CANVAS_SIZE - PLACEMENT_BORDER - size)
            box = (r, c, r + size - 1, c + size - 1)
            if all(_rect_separated(box, other) for other in boxes):
                placements[idx] = (r, c)
                boxes.append(box)
                placed = True
                break
        if not placed:
            return None
    return placements


def _sample_marker_names(count: int) -> list[str]:
    names = [
        random.choice(SLOT_FAMILIES["flip_ud"]),
        random.choice(SLOT_FAMILIES["flip_lr"]),
        random.choice(SLOT_FAMILIES["diag_main"]),
        random.choice(SLOT_FAMILIES["diag_anti"]),
    ]
    all_slots = list(SLOT_TO_OP)
    while len(names) < count:
        names.append(random.choice(all_slots))
    random.shuffle(names)
    return names


def _render_pair(specs: list[dict], placements: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((CANVAS_SIZE, CANVAS_SIZE), Color.BLACK, dtype=int)
    out = np.full((CANVAS_SIZE, CANVAS_SIZE), Color.BLACK, dtype=int)

    for spec, (r0, c0) in zip(specs, placements):
        size = spec["size"]
        color = spec["color"]
        marker = _special_slots(size)[spec["marker_name"]]

        in_patch = np.full((size, size), Color.BLACK, dtype=int)
        in_patch[spec["mask"]] = color
        in_patch[marker] = Color.YELLOW
        inp[r0 : r0 + size, c0 : c0 + size] = in_patch

        out_patch = np.full((size, size), Color.BLACK, dtype=int)
        out_patch[_apply_transform(spec["mask"], spec["op"])] = color
        out[r0 : r0 + size, c0 : c0 + size] = out_patch

    return inp, out


def _crop_canvas(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != Color.BLACK)
    r0, c0, r1, c1 = bbox_coords(coords)
    top = random.randint(1, min(3, r0))
    left = random.randint(1, min(3, c0))
    bottom = random.randint(1, min(3, inp.shape[0] - 1 - r1))
    right = random.randint(1, min(3, inp.shape[1] - 1 - c1))
    rr0 = r0 - top
    cc0 = c0 - left
    rr1 = r1 + bottom
    cc1 = c1 + right
    return inp[rr0 : rr1 + 1, cc0 : cc1 + 1].copy(), out[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()


def _validate_generated_pair(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        objects = _parse_objects(inp)
    except Exception:
        return False

    if {obj["op"] for obj in objects} != set(TRANSFORMS):
        return False

    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        patch = crop_bbox(expected, obj["bbox"])
        transformed = _apply_transform(obj["mask"], obj["op"])
        if np.any(patch == Color.YELLOW):
            return False
        if not np.all((patch == Color.BLACK) | (patch == obj["color"])):
            return False
        if not np.array_equal(patch == obj["color"], transformed):
            return False
        if np.array_equal(obj["mask"], transformed):
            return False
        variant_keys = {_array_key(_apply_transform(obj["mask"], op)) for op in TRANSFORMS}
        if len(variant_keys) != len(TRANSFORMS):
            return False

    return np.array_equal(main(inp), expected)


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*OBJECT_COUNT_RANGE)
    marker_names = _sample_marker_names(count)
    colors = random.sample(Color.OBJECT_POOL, k=count)
    specs = []

    for marker_name, color in zip(marker_names, colors):
        size = random.choices(SIZE_CHOICES, weights=(3, 2), k=1)[0]
        mask = _make_object_mask(size, marker_name)
        specs.append(
            {
                "size": size,
                "color": color,
                "mask": mask,
                "marker_name": marker_name,
                "op": SLOT_TO_OP[marker_name],
            }
        )

    placements = _place_specs(specs)
    if placements is None:
        return None

    inp, expected = _render_pair(specs, placements)
    inp, expected = _crop_canvas(inp, expected)
    if not _validate_generated_pair(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
