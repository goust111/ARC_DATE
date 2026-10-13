import random
from itertools import combinations

import numpy as np


BG_COLOR = 0
GREY_COLOR = 5
REGION_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


GRID_SIDE_MIN = 22
GRID_SIDE_MAX = 30
REGION_COUNT_RANGE = (2, 4)
HOLE_COUNT_CHOICES = (1, 2, 3)
HOLE_AREA_RANGE = (1, 5)
TEMPLATE_CANVAS_RANGE = (5, 10)
REGION_PAD_RANGE = (1, 3)
DISTRACTOR_COUNT_RANGE = (0, 2)
PLACEMENT_PADDING = 1
MAX_TEMPLATE_TRIES = 120
MAX_OBJECT_PLACE_TRIES = 120
MAX_GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))

    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    return (
        int(arr[:, 0].min()),
        int(arr[:, 1].min()),
        int(arr[:, 0].max()),
        int(arr[:, 1].max()),
    )


def normalize_coords(coords) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    arr = np.asarray(list(coords) if not isinstance(coords, np.ndarray) else coords, dtype=int)
    if arr.size == 0:
        return tuple(), (0, 0)
    mins = arr.min(axis=0)
    rel = arr - mins
    ordered = tuple(sorted((int(r), int(c)) for r, c in rel.tolist()))
    shape = tuple(int(v) for v in (rel.max(axis=0) + 1))
    return ordered, shape


def signature_to_array(signature: tuple[tuple[int, int], ...]) -> np.ndarray:
    return np.asarray(signature, dtype=int)


def region_holes(grid: np.ndarray, region_coords: np.ndarray, bg: int) -> list[np.ndarray]:
    r0, c0, r1, c1 = bbox_coords(region_coords)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    region_mask = np.zeros((h, w), dtype=bool)
    rel = region_coords - np.array([r0, c0], dtype=int)
    region_mask[rel[:, 0], rel[:, 1]] = True

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    bg_mask = (sub == bg) & (~region_mask)

    holes = []
    for comp in components_mask(bg_mask):
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 0] == h - 1):
            continue
        if np.any(comp[:, 1] == 0) or np.any(comp[:, 1] == w - 1):
            continue
        holes.append(comp + np.array([r0, c0], dtype=int))
    return holes


def extract_region_templates(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    templates: list[dict] = []

    for color in np.unique(g):
        color = int(color)
        if color in (bg, GREY_COLOR):
            continue
        for coords in components_mask(g == color):
            holes = region_holes(g, coords, bg)
            if not holes:
                continue
            union = np.concatenate(holes, axis=0)
            signature, shape = normalize_coords(union)
            templates.append(
                {
                    "color": color,
                    "signature": signature,
                    "shape": shape,
                    "hole_count": len(holes),
                    "area": len(signature),
                }
            )
    return templates


def extract_grey_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = []
    for coords in components_mask(g == GREY_COLOR):
        signature, shape = normalize_coords(coords)
        comps.append(
            {
                "coords": coords,
                "signature": signature,
                "shape": shape,
                "area": len(signature),
            }
        )
    return comps


def candidate_subsets(template: dict, grey_components: list[dict]) -> list[tuple[int, ...]]:
    indices = range(len(grey_components))
    need_count = int(template["hole_count"])
    need_area = int(template["area"])
    need_shape = template["shape"]
    need_signature = template["signature"]

    cands = []
    for combo in combinations(indices, need_count):
        if sum(grey_components[i]["area"] for i in combo) != need_area:
            continue
        union = np.concatenate([grey_components[i]["coords"] for i in combo], axis=0)
        signature, shape = normalize_coords(union)
        if shape == need_shape and signature == need_signature:
            cands.append(combo)
    return cands


def solve_assignment(
    templates: list[dict], grey_components: list[dict]
) -> tuple[list[tuple[int, ...]] | None, list[list[tuple[int, ...]]], int]:
    cand_lists = [candidate_subsets(template, grey_components) for template in templates]
    order = sorted(range(len(templates)), key=lambda i: len(cand_lists[i]))
    chosen: list[tuple[int, ...] | None] = [None] * len(templates)
    used: set[int] = set()
    solutions: list[list[tuple[int, ...] | None]] = []

    def dfs(depth: int) -> None:
        if len(solutions) > 1:
            return
        if depth == len(order):
            solutions.append(chosen.copy())
            return
        ti = order[depth]
        for combo in cand_lists[ti]:
            combo_set = set(combo)
            if used & combo_set:
                continue
            chosen[ti] = combo
            used.update(combo_set)
            dfs(depth + 1)
            used.difference_update(combo_set)
            chosen[ti] = None

    dfs(0)
    assignment = None
    if len(solutions) == 1:
        assignment = [tuple(combo) for combo in solutions[0] if combo is not None]
        if len(assignment) != len(templates):
            assignment = None
    return assignment, cand_lists, len(solutions)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each region's enclosed hole pattern as a template, and recolor the uniquely
    matching grey placeholder pattern with that region's color.
    """
    g = np.asarray(grid, dtype=int)
    templates = extract_region_templates(g)
    if not templates:
        return g.copy()

    grey_components = extract_grey_components(g)
    assignment, _cand_lists, solution_count = solve_assignment(templates, grey_components)
    assert assignment is not None and solution_count == 1

    out = g.copy()
    for template, combo in zip(templates, assignment):
        for idx in combo:
            coords = grey_components[idx]["coords"]
            out[coords[:, 0], coords[:, 1]] = int(template["color"])
    return out


def random_polyomino(area: int, max_h: int = 4, max_w: int = 4) -> tuple[tuple[int, int], ...]:
    for _ in range(80):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = {}
            for r, c in cells:
                for dr, dc in DIR4:
                    nxt = (r + dr, c + dc)
                    if nxt in cells:
                        continue
                    score = sum((nxt[0] + rr, nxt[1] + cc) in cells for rr, cc in DIR4)
                    frontier[nxt] = frontier.get(nxt, 0) + score
            choices = list(frontier)
            weights = [frontier[cell] for cell in choices]
            cells.add(random.choices(choices, weights=weights, k=1)[0])
        signature, shape = normalize_coords(cells)
        if shape[0] <= max_h and shape[1] <= max_w:
            return signature
    raise RuntimeError("failed to generate compact polyomino")


def build_template_components(num_holes: int) -> list[tuple[tuple[int, int], ...]]:
    for _ in range(MAX_TEMPLATE_TRIES):
        shapes = [
            random_polyomino(random.randint(*HOLE_AREA_RANGE), max_h=3, max_w=3)
            for _ in range(num_holes)
        ]
        canvas_h = random.randint(*TEMPLATE_CANVAS_RANGE)
        canvas_w = random.randint(*TEMPLATE_CANVAS_RANGE)
        occupied = set()
        blocked = set()
        mask = np.zeros((canvas_h, canvas_w), dtype=bool)
        ok = True

        order = list(range(num_holes))
        random.shuffle(order)
        for idx in order:
            shape = shapes[idx]
            shape_arr = signature_to_array(shape)
            sh = int(shape_arr[:, 0].max()) + 1
            sw = int(shape_arr[:, 1].max()) + 1
            placements = []
            for r0 in range(canvas_h - sh + 1):
                for c0 in range(canvas_w - sw + 1):
                    abs_cells = [(r0 + r, c0 + c) for r, c in shape]
                    if any(cell in blocked for cell in abs_cells):
                        continue
                    placements.append(abs_cells)
            if not placements:
                ok = False
                break
            cells = random.choice(placements)
            for r, c in cells:
                mask[r, c] = True
                occupied.add((r, c))
            for r, c in cells:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < canvas_h and 0 <= nc < canvas_w:
                            blocked.add((nr, nc))

        if not ok or not occupied:
            continue

        union_coords = np.argwhere(mask)
        signature, shape = normalize_coords(union_coords)
        tight = np.zeros(shape, dtype=bool)
        sig_arr = signature_to_array(signature)
        tight[sig_arr[:, 0], sig_arr[:, 1]] = True
        comps = components_mask(tight)
        if len(comps) != num_holes:
            continue
        return [tuple(sorted((int(r), int(c)) for r, c in comp.tolist())) for comp in comps]

    raise RuntimeError("failed to build hole template")


def build_region_spec(color: int, hole_count: int) -> dict:
    components = build_template_components(hole_count)
    union = np.concatenate([signature_to_array(comp) for comp in components], axis=0)
    signature, template_shape = normalize_coords(union)
    top_pad = random.randint(*REGION_PAD_RANGE)
    bottom_pad = random.randint(*REGION_PAD_RANGE)
    left_pad = random.randint(*REGION_PAD_RANGE)
    right_pad = random.randint(*REGION_PAD_RANGE)
    patch = np.full(
        (template_shape[0] + top_pad + bottom_pad, template_shape[1] + left_pad + right_pad),
        color,
        dtype=int,
    )
    for comp in components:
        arr = signature_to_array(comp)
        patch[arr[:, 0] + top_pad, arr[:, 1] + left_pad] = BG_COLOR
    return {
        "color": color,
        "hole_count": hole_count,
        "signature": signature,
        "template_shape": template_shape,
        "patch": patch,
        "area": len(signature),
    }


def can_place_bbox(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    H, W = reserved.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rs = max(0, r0 - pad)
    re = min(H, r0 + h + pad)
    cs = max(0, c0 - pad)
    ce = min(W, c0 + w + pad)
    return not reserved[rs:re, cs:ce].any()


def reserve_bbox(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    H, W = reserved.shape
    rs = max(0, r0 - pad)
    re = min(H, r0 + h + pad)
    cs = max(0, c0 - pad)
    ce = min(W, c0 + w + pad)
    reserved[rs:re, cs:ce] = True


def place_patch(grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray, pad: int) -> bool:
    h, w = patch.shape
    H, W = grid.shape
    positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(positions)
    for r0, c0 in positions:
        if not can_place_bbox(reserved, r0, c0, h, w, pad):
            continue
        grid[r0 : r0 + h, c0 : c0 + w] = patch
        reserve_bbox(reserved, r0, c0, h, w, pad)
        return True
    return False


def place_signature(
    grid: np.ndarray,
    reserved: np.ndarray,
    signature: tuple[tuple[int, int], ...],
    shape: tuple[int, int],
    color: int,
    pad: int,
) -> bool:
    h, w = shape
    H, W = grid.shape
    positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(positions)
    coords = signature_to_array(signature)
    for r0, c0 in positions:
        if not can_place_bbox(reserved, r0, c0, h, w, pad):
            continue
        grid[r0 + coords[:, 0], c0 + coords[:, 1]] = color
        reserve_bbox(reserved, r0, c0, h, w, pad)
        return True
    return False


def build_distractor(template_areas: set[int], template_signatures: set[tuple[tuple[int, int], ...]]) -> dict:
    for _ in range(80):
        area_choices = [a for a in range(HOLE_AREA_RANGE[0], HOLE_AREA_RANGE[1] + 1) if a not in template_areas]
        area = random.choice(area_choices or list(range(HOLE_AREA_RANGE[0], HOLE_AREA_RANGE[1] + 1)))
        signature = random_polyomino(area, max_h=4, max_w=4)
        if signature in template_signatures:
            continue
        _, shape = normalize_coords(signature)
        return {"signature": signature, "shape": shape}
    raise RuntimeError("failed to build distractor")


def validate_generated_grid(grid: np.ndarray, expected_pairs: set[tuple[int, tuple[tuple[int, int], ...]]]) -> bool:
    if get_bg(grid) != BG_COLOR:
        return False
    templates = extract_region_templates(grid)
    if {(tpl["color"], tpl["signature"]) for tpl in templates} != expected_pairs:
        return False
    if len({tpl["signature"] for tpl in templates}) != len(templates):
        return False

    grey_components = extract_grey_components(grid)
    assignment, cand_lists, solution_count = solve_assignment(templates, grey_components)
    if assignment is None or solution_count != 1:
        return False
    if any(len(cands) != 1 for cands in cand_lists):
        return False

    used = set()
    for combo in assignment:
        combo_set = set(combo)
        if used & combo_set:
            return False
        used.update(combo_set)
    return True


def _gen_attempt() -> np.ndarray | None:
    region_count = random.randint(*REGION_COUNT_RANGE)
    colors = random.sample(REGION_COLORS, region_count)
    hole_counts = [random.choice(HOLE_COUNT_CHOICES) for _ in range(region_count)]

    region_specs = []
    seen_signatures = set()
    for color, hole_count in zip(colors, hole_counts):
        spec = build_region_spec(color, hole_count)
        if spec["signature"] in seen_signatures:
            return None
        seen_signatures.add(spec["signature"])
        region_specs.append(spec)

    h = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    w = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    grid = np.full((h, w), BG_COLOR, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    for spec in sorted(region_specs, key=lambda item: item["patch"].size, reverse=True):
        if not place_patch(grid, reserved, spec["patch"], PLACEMENT_PADDING):
            return None

    for spec in sorted(
        region_specs, key=lambda item: item["template_shape"][0] * item["template_shape"][1], reverse=True
    ):
        if not place_signature(
            grid, reserved, spec["signature"], spec["template_shape"], GREY_COLOR, PLACEMENT_PADDING
        ):
            return None

    distractor_count = random.randint(*DISTRACTOR_COUNT_RANGE)
    template_areas = {spec["area"] for spec in region_specs}
    template_signatures = {spec["signature"] for spec in region_specs}
    for _ in range(distractor_count):
        distractor = build_distractor(template_areas, template_signatures)
        if not place_signature(
            grid, reserved, distractor["signature"], distractor["shape"], GREY_COLOR, PLACEMENT_PADDING
        ):
            return None

    expected_pairs = {(spec["color"], spec["signature"]) for spec in region_specs}
    if not validate_generated_grid(grid, expected_pairs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
