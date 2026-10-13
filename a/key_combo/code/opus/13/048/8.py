import random
from collections import deque

import numpy as np


BACKGROUND = 0
TEMPLATE_COLOR = 7
OBJECT_COLORS = (1, 2, 3, 4, 6)
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 300


SHAPE_LIBRARY = tuple(
    np.array(mask, dtype=bool)
    for mask in (
        ((1, 1), (1, 0)),
        ((1, 1), (0, 1)),
        ((1, 0), (1, 1)),
        ((0, 1), (1, 1)),
        ((1, 1, 1), (0, 1, 0)),
        ((1, 1, 0), (0, 1, 1)),
        ((1, 0), (1, 1), (0, 1)),
        ((1, 1, 1), (1, 0, 0)),
        ((1, 1, 1), (0, 0, 1)),
        ((1, 1, 1), (0, 1, 1)),
        ((1, 0, 1), (1, 1, 1)),
    )
)


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    if connectivity != 4:
        raise ValueError("only 4-connectivity is used in this task")

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def shape_key(coords: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    offsets = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return (int(r1 - r0 + 1), int(c1 - c0 + 1)), offsets


def mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    return shape_key(np.argwhere(mask))


def render_mask(mask: np.ndarray, color: int) -> np.ndarray:
    arr = np.zeros(mask.shape, dtype=int)
    arr[mask] = int(color)
    return arr


def extract_holes(panel: np.ndarray) -> list[dict]:
    holes = []
    hole_mask = panel == BACKGROUND
    for coords in components_mask(hole_mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        touches_border = r0 == 0 or c0 == 0 or r1 == panel.shape[0] - 1 or c1 == panel.shape[1] - 1
        if touches_border:
            continue
        coord_set = {tuple(cell) for cell in coords.tolist()}
        enclosed = True
        for r, c in coord_set:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in coord_set:
                    continue
                if panel[nr, nc] != TEMPLATE_COLOR:
                    enclosed = False
                    break
            if not enclosed:
                break
        if not enclosed:
            continue
        holes.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "key": shape_key(coords),
                "mask": (panel[r0 : r1 + 1, c0 : c1 + 1] == BACKGROUND),
            }
        )
    holes.sort(key=lambda hole: (hole["bbox"][0], hole["bbox"][1]))
    return holes


def extract_objects(grid: np.ndarray, template_right: int) -> list[dict]:
    objects = []
    for color in sorted(set(map(int, np.unique(grid))) - {BACKGROUND, TEMPLATE_COLOR}):
        for coords in components_mask(grid == color, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(coords)
            assert c0 > template_right, "movable objects must be to the right of the template"
            objects.append(
                {
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "key": shape_key(coords),
                    "color": int(color),
                }
            )
    objects.sort(key=lambda obj: (obj["bbox"][1], obj["bbox"][0], obj["color"]))
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the right-side objects left and embed only the ones whose shape matches an enclosed template hole at the same top row.

    Input structure:
    - Black (0) background surrounds one orange (7) template on the left and several small colored objects on the right.
    - The template contains 2-3 enclosed black holes; each hole is a 4-connected irregular region.
    - The right-side objects are monochromatic 4-connected sprites in colors 1, 2, 3, 4, or 6.

    Rule:
    - Extract the template's enclosed black holes and normalize each hole mask to its bounding box.
    - Extract every colored right-side object and normalize its mask to its bounding box.
    - Process objects from nearest to farthest from the template's right edge.
    - When an object reaches the template, it fills the unique still-empty hole whose normalized mask matches the object and whose hole bbox top row equals the object's bbox top row.
    - Objects that do not satisfy both tests keep sliding left and disappear; the output keeps only the left template area with any filled holes recolored by the matching object color.

    Key insight:
    - The slide matters only because earlier objects claim holes first, while the actual docking test is a joint signature of shape and vertical alignment rather than a simple collision stop.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    template_coords = np.argwhere(g == TEMPLATE_COLOR)
    assert template_coords.size > 0, "expected an orange template"
    _r0, _c0, _r1, template_right = bbox_coords(template_coords)
    panel = g[:, : template_right + 1]
    out = panel.copy()

    holes = extract_holes(panel)
    hole_by_signature: dict[tuple[tuple[tuple[int, int], tuple[tuple[int, int], ...]], int], int] = {}
    for idx, hole in enumerate(holes):
        signature = (hole["key"], hole["bbox"][0])
        assert signature not in hole_by_signature, "hole signatures must be unique"
        hole_by_signature[signature] = idx

    filled = [False] * len(holes)
    for obj in extract_objects(g, template_right):
        signature = (obj["key"], obj["bbox"][0])
        hole_idx = hole_by_signature.get(signature)
        if hole_idx is None or filled[hole_idx]:
            continue
        coords = holes[hole_idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = obj["color"]
        filled[hole_idx] = True
    return out


def _choose_hole_masks(hole_count: int) -> list[np.ndarray]:
    repeated = random.choice(SHAPE_LIBRARY)
    if hole_count == 2:
        masks = [repeated.copy(), repeated.copy()]
    else:
        other = random.choice([mask for mask in SHAPE_LIBRARY if mask_key(mask) != mask_key(repeated)])
        masks = [repeated.copy(), repeated.copy(), other.copy()]
    random.shuffle(masks)
    return masks


def _build_left_panel(hole_count: int) -> tuple[np.ndarray, list[dict], int]:
    template_width = random.randint(6, 8)
    template_height = 4 * hole_count + 3
    height = random.randint(template_height + 2, min(22, template_height + 6))
    template_top = random.randint(1, height - template_height - 1)
    template_bottom = template_top + template_height - 1

    panel = np.zeros((height, template_width), dtype=int)
    panel[template_top : template_bottom + 1, :] = TEMPLATE_COLOR

    holes = []
    masks = _choose_hole_masks(hole_count)
    for idx, mask in enumerate(masks):
        mh, mw = mask.shape
        band_top = template_top + 1 + 4 * idx
        top_choices = list(range(band_top, band_top + 4 - mh + 1))
        left_choices = list(range(1, template_width - mw))
        top = random.choice(top_choices)
        left = random.choice(left_choices)
        coords = np.argwhere(mask) + np.array([top, left], dtype=int)
        panel[coords[:, 0], coords[:, 1]] = BACKGROUND
        holes.append(
            {
                "coords": coords,
                "bbox": (top, left, top + mh - 1, left + mw - 1),
                "key": mask_key(mask),
                "mask": mask.copy(),
            }
        )
    holes.sort(key=lambda hole: (hole["bbox"][0], hole["bbox"][1]))
    return panel, holes, template_width


def _make_case_plan(hole_count: int) -> tuple[list[int], int | None]:
    if hole_count == 2:
        if random.random() < 0.5:
            return [0, 1], None
        target = random.randrange(2)
        return [target], target

    if random.random() < 0.5:
        return [0, 1, 2], None
    unmatched = random.randrange(3)
    matched = [idx for idx in range(3) if idx != unmatched]
    duplicate_target = random.choice(matched)
    return matched, duplicate_target


def _can_place(mask: np.ndarray, top: int, left: int, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    coords = np.argwhere(mask)
    for dr, dc in coords:
        r = top + int(dr)
        c = left + int(dc)
        if occupied[r, c]:
            return False
        r0 = max(0, r - 1)
        r1 = min(h - 1, r + 1)
        c0 = max(0, c - 1)
        c1 = min(w - 1, c + 1)
        if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
            return False
    return True


def _place_mask(mask: np.ndarray, top: int, left: int, occupied: np.ndarray) -> None:
    coords = np.argwhere(mask)
    occupied[top + coords[:, 0], left + coords[:, 1]] = True


def _build_object_specs(holes: list[dict], matched_holes: list[int], duplicate_target: int | None) -> list[dict]:
    total_objects = len(holes) + 2
    colors = random.sample(list(OBJECT_COLORS), total_objects)
    specs = []
    color_idx = 0

    for hole_idx in matched_holes:
        hole = holes[hole_idx]
        specs.append(
            {
                "kind": "match",
                "hole_idx": hole_idx,
                "mask": hole["mask"].copy(),
                "key": hole["key"],
                "top": hole["bbox"][0],
                "color": colors[color_idx],
            }
        )
        color_idx += 1

    if duplicate_target is not None:
        hole = holes[duplicate_target]
        specs.append(
            {
                "kind": "duplicate",
                "hole_idx": duplicate_target,
                "mask": hole["mask"].copy(),
                "key": hole["key"],
                "top": hole["bbox"][0],
                "color": colors[color_idx],
            }
        )
        color_idx += 1

    hole_keys = {hole["key"] for hole in holes}
    decoy_shapes = [mask for mask in SHAPE_LIBRARY if mask_key(mask) not in hole_keys]
    while len(specs) < total_objects:
        target_hole = random.choice(holes)
        mask = random.choice(decoy_shapes).copy()
        specs.append(
            {
                "kind": "decoy",
                "hole_idx": None,
                "mask": mask,
                "key": mask_key(mask),
                "top": target_hole["bbox"][0],
                "color": colors[color_idx],
            }
        )
        color_idx += 1

    random.shuffle(specs)
    return specs


def _place_objects(height: int, template_width: int, specs: list[dict]) -> tuple[int, list[dict]] | None:
    spec_order = sorted(
        range(len(specs)),
        key=lambda idx: (specs[idx]["kind"] == "duplicate", specs[idx]["kind"] == "decoy"),
    )
    for _ in range(PLACEMENT_ATTEMPTS):
        width = random.randint(template_width + 10, min(26, template_width + 16))
        occupied = np.zeros((height, width), dtype=bool)
        placed: list[dict | None] = [None] * len(specs)
        used_lefts: set[int] = set()
        match_left_by_hole: dict[int, int] = {}
        ok = True

        for idx in spec_order:
            spec = specs[idx]
            mh, mw = spec["mask"].shape
            min_left = template_width + 2
            if spec["kind"] == "duplicate":
                base_left = match_left_by_hole[spec["hole_idx"]]
                min_left = base_left + 1
            max_left = width - mw
            if min_left > max_left:
                ok = False
                break
            candidates = list(range(min_left, max_left + 1))
            random.shuffle(candidates)
            placed_here = None
            for left in candidates:
                if left in used_lefts:
                    continue
                if not _can_place(spec["mask"], spec["top"], left, occupied):
                    continue
                _place_mask(spec["mask"], spec["top"], left, occupied)
                used_lefts.add(left)
                placed_here = dict(spec)
                placed_here["left"] = left
                if spec["kind"] == "match":
                    match_left_by_hole[spec["hole_idx"]] = left
                break
            if placed_here is None:
                ok = False
                break
            placed[idx] = placed_here

        if ok:
            return width, [item for item in placed if item is not None]
    return None


def _render_case(panel: np.ndarray, holes: list[dict], placed_objects: list[dict], width: int) -> tuple[np.ndarray, np.ndarray]:
    height = panel.shape[0]
    inp = np.zeros((height, width), dtype=int)
    inp[:, : panel.shape[1]] = panel
    out = panel.copy()

    for obj in placed_objects:
        sprite = render_mask(obj["mask"], obj["color"])
        mh, mw = sprite.shape
        top = obj["top"]
        left = obj["left"]
        inp[top : top + mh, left : left + mw][sprite != 0] = obj["color"]

    for obj in placed_objects:
        if obj["kind"] != "match":
            continue
        coords = holes[obj["hole_idx"]]["coords"]
        out[coords[:, 0], coords[:, 1]] = obj["color"]
    return inp, out


def _validate_generated_case(inp: np.ndarray, out: np.ndarray, holes: list[dict], placed_objects: list[dict], duplicate_target: int | None) -> bool:
    solved = main(inp)
    if not np.array_equal(solved, out):
        return False

    template_right = np.argwhere(inp == TEMPLATE_COLOR)[:, 1].max()
    found_holes = extract_holes(inp[:, : template_right + 1])
    found_objects = extract_objects(inp, template_right)
    if len(found_holes) != len(holes) or len(found_objects) != len(placed_objects):
        return False

    if len({obj["left"] for obj in placed_objects}) != len(placed_objects):
        return False

    if duplicate_target is not None:
        dup = next(obj for obj in placed_objects if obj["kind"] == "duplicate")
        winner = next(obj for obj in placed_objects if obj["kind"] == "match" and obj["hole_idx"] == duplicate_target)
        if dup["left"] <= winner["left"]:
            return False
        hole_coords = holes[duplicate_target]["coords"]
        if np.unique(out[hole_coords[:, 0], hole_coords[:, 1]]).tolist() != [winner["color"]]:
            return False
        if dup["color"] in out:
            return False

    for obj in placed_objects:
        if obj["kind"] == "decoy" and obj["color"] in out:
            return False

    hole_shape_counts: dict[tuple[tuple[int, int], tuple[tuple[int, int], ...]], list[int]] = {}
    for hole in holes:
        hole_shape_counts.setdefault(hole["key"], []).append(hole["bbox"][0])
    if max(len(rows) for rows in hole_shape_counts.values()) < 2:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    hole_count = random.choice((2, 3))
    panel, holes, template_width = _build_left_panel(hole_count)
    matched_holes, duplicate_target = _make_case_plan(hole_count)
    specs = _build_object_specs(holes, matched_holes, duplicate_target)
    placement = _place_objects(panel.shape[0], template_width, specs)
    if placement is None:
        return None
    width, placed_objects = placement
    inp, out = _render_case(panel, holes, placed_objects, width)
    if not _validate_generated_case(inp, out, holes, placed_objects, duplicate_target):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
