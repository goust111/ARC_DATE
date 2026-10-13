import random
from collections import Counter
from itertools import permutations

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def bbox_of_coords(coords) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return bbox_of_coords((tuple(map(int, rc)) for rc in coords))


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def bottom_left_anchor(coords) -> tuple[int, int]:
    pts = list(coords)
    bottom = max(r for r, _ in pts)
    left = min(c for r, c in pts if r == bottom)
    return bottom, left


def normalize_coords(coords):
    pts = list(coords)
    r0, c0, _r1, _c1 = bbox_of_coords(pts)
    return {(r - r0, c - c0) for r, c in pts}


def paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int = 0) -> None:
    h, w = patch.shape
    for r in range(h):
        for c in range(w):
            val = int(patch[r, c])
            if val != transparent:
                grid[r0 + r, c0 + c] = val


def make_patch(coords, color: int) -> tuple[np.ndarray, set[tuple[int, int]]]:
    pts = list(coords)
    r0, c0, r1, c1 = bbox_of_coords(pts)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    normalized = set()
    for r, c in pts:
        rr = r - r0
        cc = c - c0
        patch[rr, cc] = color
        normalized.add((rr, cc))
    return patch, normalized


def shear_relative_coords(source_coords, sign: int) -> set[tuple[int, int]]:
    anchor_r, anchor_c = bottom_left_anchor(source_coords)
    rels = set()
    for r, c in source_coords:
        dist = anchor_r - r
        rels.add((-dist, c - anchor_c + sign * dist))
    return rels


def transformed_coords(source_coords, sign: int) -> set[tuple[int, int]]:
    anchor_r, _anchor_c = bottom_left_anchor(source_coords)
    out = set()
    for r, c in source_coords:
        dist = anchor_r - r
        out.add((r, c + sign * dist))
    return out


def solve_candidates(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = sorted(set(map(int, g.ravel())) - {bg})
    assert len(colors) == 3, "expected exactly three foreground colors"

    candidates: list[dict] = []
    seen_keys = set()
    for source_color in colors:
        source_mask = g == source_color
        source_comps = components_mask(source_mask)
        if len(source_comps) != 1:
            continue
        source_coords = set(source_comps[0])
        source_bbox = bbox_of_coords(source_coords)

        other = [c for c in colors if c != source_color]
        for shadow_color, damage_color in permutations(other, 2):
            shadow_coords = {
                (int(r), int(c))
                for r, c in np.argwhere(g == shadow_color)
            }
            damage_coords = {
                (int(r), int(c))
                for r, c in np.argwhere(g == damage_color)
            }
            if not shadow_coords or not damage_coords:
                continue

            observed_mask = (g == shadow_color) | (g == damage_color)
            observed_bbox = bbox_of_mask(observed_mask)
            if observed_bbox is None:
                continue
            r0, c0, r1, c1 = observed_bbox
            damage_in_bbox = {
                cell for cell in damage_coords if r0 <= cell[0] <= r1 and c0 <= cell[1] <= c1
            }
            if damage_in_bbox != damage_coords:
                continue

            for sign in (-1, 1):
                rel_coords = shear_relative_coords(source_coords, sign)
                rel_r0, rel_c0, rel_r1, rel_c1 = bbox_of_coords(rel_coords)
                assert rel_r1 == 0
                shadow_anchor = (r1, c0 - rel_c0)
                predicted = {
                    (shadow_anchor[0] + rr, shadow_anchor[1] + cc)
                    for rr, cc in rel_coords
                }
                if not all(in_bounds(g.shape, r, c) for r, c in predicted):
                    continue
                if bbox_of_coords(predicted) != observed_bbox:
                    continue
                if not shadow_coords.issubset(predicted):
                    continue
                if any((r, c) in source_coords for r, c in predicted):
                    continue
                if not (damage_coords & predicted):
                    continue
                if source_bbox == observed_bbox:
                    continue

                key = (source_color, shadow_color, damage_color, sign, shadow_anchor)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                candidates.append(
                    {
                        "bg": bg,
                        "source_color": source_color,
                        "shadow_color": shadow_color,
                        "damage_color": damage_color,
                        "sign": sign,
                        "source_coords": source_coords,
                        "shadow_coords": predicted,
                        "damage_coords": damage_coords,
                        "observed_bbox": observed_bbox,
                        "shadow_anchor": shadow_anchor,
                    }
                )
    return candidates


def solve_instance(grid: np.ndarray) -> dict:
    candidates = solve_candidates(grid)
    assert len(candidates) == 1, "expected a unique source/shadow/shear interpretation"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the damaged sheared twin by mapping each damaged shadow-row cell back to the intact source object.

    Input structure:
    - The background is dominant, with exactly three foreground colors: source, shadow, and damage.
    - One color forms a single intact 4-connected source object.
    - Another color, together with the damage color, occupies the shadow's bounding box: the shadow is a row-wise shear of the source, and some shadow or in-bbox background cells are overwritten by damage.

    Rule:
    - Identify the intact source object and the damaged shadow bbox.
    - Using each object's bottom-row leftmost pixel as anchor, infer whether the shadow shears left or right by one extra cell per row above the bottom.
    - For every damage pixel, if its position belongs to the inferred shadow shape, recolor it to the shadow color; otherwise restore it to background.
    - Leave the source unchanged and remove the damage color entirely.

    Key insight:
    - The two objects are linked by a non-rigid bottom-anchored row mapping, so repair depends on reconstructing the shear direction and then querying the corresponding source row pattern.
    """
    g = np.asarray(grid, dtype=int)
    solution = solve_instance(g)
    out = g.copy()
    shadow_coords = solution["shadow_coords"]
    shadow_color = solution["shadow_color"]
    bg = solution["bg"]
    for r, c in solution["damage_coords"]:
        out[r, c] = shadow_color if (r, c) in shadow_coords else bg
    return out


def generate_source_coords(rng: random.Random) -> set[tuple[int, int]] | None:
    for _ in range(200):
        h = rng.randint(4, 7)
        w = rng.randint(4, 7)
        max_area = min(h * w - 1, int(0.72 * h * w) + 1)
        min_area = max(h + 3, 8)
        if min_area > max_area:
            continue
        target = rng.randint(min_area, max_area)

        spine_cols = [rng.randrange(w)]
        for _row in range(1, h):
            prev = spine_cols[-1]
            options = [c for c in (prev - 1, prev, prev + 1) if 0 <= c < w]
            spine_cols.append(rng.choice(options))

        coords = {(r, c) for r, c in enumerate(spine_cols)}
        if w >= 2:
            extra_bottom = rng.choice([c for c in range(w) if c != spine_cols[-1]])
            coords.add((h - 1, extra_bottom))

        while len(coords) < target:
            frontier = set()
            for r, c in coords:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in coords:
                        frontier.add((nr, nc))
            if not frontier:
                break
            coords.add(rng.choice(tuple(frontier)))
        if len(coords) != target:
            continue

        rows = {r: [] for r in range(h)}
        for r, c in coords:
            rows[r].append(c)
        if any(not rows[r] for r in range(h)):
            continue
        if len(rows[h - 1]) < 2:
            continue

        row_profiles = {(min(cols), max(cols), len(cols)) for cols in rows.values()}
        if len(row_profiles) < 3:
            continue

        left_shape = normalize_coords(transformed_coords(coords, -1))
        right_shape = normalize_coords(transformed_coords(coords, 1))
        if left_shape == right_shape:
            continue

        if len(components_mask(mask_from_coords(coords))) != 1:
            continue
        return coords
    return None


def mask_from_coords(coords) -> np.ndarray:
    pts = list(coords)
    r0, c0, r1, c1 = bbox_of_coords(pts)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        mask[r - r0, c - c0] = True
    return mask


def build_shadow_variants(source_coords, shadow_color: int, sign: int):
    anchor_r, _anchor_c = bottom_left_anchor(source_coords)
    transformed = set()
    for r, c in source_coords:
        dist = anchor_r - r
        transformed.add((r, c + sign * dist))
    return make_patch(transformed, shadow_color)


def choose_damage_cells(
    shadow_patch: np.ndarray,
    shadow_coords: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    h, w = shadow_patch.shape
    bg_cells = {(r, c) for r in range(h) for c in range(w) if (r, c) not in shadow_coords}
    if len(bg_cells) < 1:
        return None

    shadow_rows = {}
    for r, c in shadow_coords:
        shadow_rows.setdefault(r, []).append(c)
    mandatory_visible = {(0, min(shadow_rows[0])), (h - 1, min(shadow_rows[h - 1]))}

    shadow_candidates = list(shadow_coords - mandatory_visible)
    if len(shadow_candidates) < 2:
        return None

    for _ in range(80):
        max_shadow_damage = min(len(shadow_candidates), max(2, len(shadow_coords) // 3))
        k_shadow = rng.randint(1, max_shadow_damage)
        on_shape = set(rng.sample(shadow_candidates, k_shadow))
        visible = shadow_coords - on_shape
        if len({r for r, _ in visible}) < 3:
            continue

        k_bg = rng.randint(1, min(4, len(bg_cells)))
        off_shape = set(rng.sample(list(bg_cells), k_bg))
        if len(on_shape) >= 1 and len(off_shape) >= 1:
            return on_shape, off_shape
    return None


def place_boxes(
    grid_shape: tuple[int, int],
    source_shape: tuple[int, int],
    shadow_shape: tuple[int, int],
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = grid_shape
    hs, ws = source_shape
    ht, wt = shadow_shape
    source_positions = [
        (r, c)
        for r in range(1, h - hs)
        for c in range(1, w - ws)
    ]
    rng.shuffle(source_positions)
    for sr, sc in source_positions:
        shadow_positions = []
        for tr in range(1, h - ht):
            for tc in range(1, w - wt):
                if tr + ht < sr - 1 or sr + hs < tr - 1 or tc + wt < sc - 1 or sc + ws < tc - 1:
                    shadow_positions.append((tr, tc))
        if shadow_positions:
            return (sr, sc), rng.choice(shadow_positions)
    return None


def build_input_output(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    source_coords = generate_source_coords(rng)
    if source_coords is None:
        return None

    bg = 0
    source_color, shadow_color, damage_color = rng.sample(range(1, 10), 3)
    source_patch, normalized_source = make_patch(source_coords, source_color)
    sign = rng.choice([-1, 1])
    shadow_patch, normalized_shadow = build_shadow_variants(normalized_source, shadow_color, sign)

    damage_choice = choose_damage_cells(shadow_patch, normalized_shadow, rng)
    if damage_choice is None:
        return None
    on_shape, off_shape = damage_choice

    shadow_input = shadow_patch.copy()
    for r, c in on_shape | off_shape:
        shadow_input[r, c] = damage_color

    source_area = int(np.count_nonzero(source_patch))
    shadow_area = int(np.count_nonzero(shadow_input))
    for _ in range(80):
        grid_h = rng.randint(max(source_patch.shape[0], shadow_patch.shape[0]) + 4, 20)
        grid_w = rng.randint(max(source_patch.shape[1], shadow_patch.shape[1]) + 6, 24)
        if source_area + shadow_area >= (grid_h * grid_w) // 2:
            continue
        placement = place_boxes((grid_h, grid_w), source_patch.shape, shadow_patch.shape, rng)
        if placement is None:
            continue
        (sr, sc), (tr, tc) = placement
        inp = np.zeros((grid_h, grid_w), dtype=int)
        out = np.zeros((grid_h, grid_w), dtype=int)
        paint_patch(inp, source_patch, sr, sc)
        paint_patch(inp, shadow_input, tr, tc)
        paint_patch(out, source_patch, sr, sc)
        paint_patch(out, shadow_patch, tr, tc)
        return inp, out
    return None


def validate_scene(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        candidates = solve_candidates(inp)
        if len(candidates) != 1:
            return False
        candidate = candidates[0]
        damage_coords = candidate["damage_coords"]
        shadow_coords = candidate["shadow_coords"]
        if not damage_coords:
            return False
        if not any(cell in shadow_coords for cell in damage_coords):
            return False
        if not any(cell not in shadow_coords for cell in damage_coords):
            return False
        solved = main(inp)
        if not np.array_equal(solved, out):
            return False
        if candidate["damage_color"] in solved:
            return False
        if len(set(map(int, inp.ravel())) - {get_bg(inp)}) != 3:
            return False
        if len(components_mask(inp == candidate["source_color"])) != 1:
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        scene = build_input_output(rng)
        if scene is None:
            continue
        inp, out = scene
        if validate_scene(inp, out):
            return inp
    raise RuntimeError("failed to generate a valid剪切孪生 input")
