from __future__ import annotations

import numpy as np


BG_COLOR = 8
SHADOW_COLOR = 5
ENTITY_COLORS = (0, 1, 2, 3, 4, 6, 7, 9)

GRID_SIDE_MIN = 14
GRID_SIDE_MAX = 21
ENTITY_COUNT_MIN = 4
ENTITY_COUNT_MAX = 7
SIZE_LIMIT = 4
PLACEMENT_GAP = 1
OFFSET_VALUES = tuple(
    (dr, dc)
    for dr in range(-3, 4)
    for dc in range(-3, 4)
    if (dr, dc) != (0, 0) and max(abs(dr), abs(dc)) >= 2
)
GENERATION_RETRIES = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue

            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )

    return components


def _shape_signature(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    shifted = np.asarray(coords, dtype=int) - np.min(coords, axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _bbox_size(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _component_key(component: dict) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, _r1, _c1 = component["bbox"]
    return int(r0), int(c0), _shape_signature(component["coords"])


def _infer_offset(entities: list[dict], shadows: list[dict]) -> tuple[int, int]:
    shadow_keys = {_component_key(comp) for comp in shadows}
    candidates: set[tuple[int, int]] = set()

    for entity in entities:
        er0, ec0, _er1, _ec1 = entity["bbox"]
        esig = _shape_signature(entity["coords"])
        for shadow in shadows:
            sr0, sc0, _sr1, _sc1 = shadow["bbox"]
            if esig == _shape_signature(shadow["coords"]):
                delta = (int(sr0 - er0), int(sc0 - ec0))
                if delta != (0, 0):
                    candidates.add(delta)

    assert candidates, "expected at least one entity-shadow pair"

    scored: list[tuple[bool, int, tuple[int, int]]] = []
    for dr, dc in candidates:
        matched = 0
        covered_shadows: set[tuple[int, int, tuple[tuple[int, int], ...]]] = set()
        for entity in entities:
            er0, ec0, _er1, _ec1 = entity["bbox"]
            key = (int(er0 + dr), int(ec0 + dc), _shape_signature(entity["coords"]))
            if key in shadow_keys:
                matched += 1
                covered_shadows.add(key)
        scored.append((len(covered_shadows) == len(shadows), matched, (dr, dc)))

    best_score = max((flag, matched) for flag, matched, _offset in scored)
    best = [offset for flag, matched, offset in scored if (flag, matched) == best_score]
    assert len(best) == 1, "shadow offset must be uniquely inferable"
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the global shadow offset from existing entity-shadow pairs, then add missing shadows.

    Input structure
    - Background is azure.
    - Non-grey objects are 4-connected monochrome solid rectangles.
    - Some rectangles have a same-shape grey shadow at a consistent offset.

    Rule
    - Infer the shared offset vector (dr, dc) from the visible entity-shadow pairs.
    - For every entity without a shadow at that offset, draw a grey shadow there.
    - Shadows sit below entities, so overlapping entity pixels stay unchanged.

    Key insight
    Existing shadows reveal a single global translation; once that translation is known,
    the same shadow can be completed for every unpaired entity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    components = _find_components(g, bg)
    entities = [comp for comp in components if int(comp["color"]) != SHADOW_COLOR]
    shadows = [comp for comp in components if int(comp["color"]) == SHADOW_COLOR]

    assert entities, "expected at least one entity"
    assert shadows, "expected at least one visible shadow"

    dr, dc = _infer_offset(entities, shadows)
    shadow_keys = {_component_key(comp) for comp in shadows}

    entity_mask = np.zeros(g.shape, dtype=bool)
    for entity in entities:
        coords = entity["coords"]
        entity_mask[coords[:, 0], coords[:, 1]] = True

    out = g.copy()
    h, w = g.shape
    for entity in entities:
        er0, ec0, _er1, _ec1 = entity["bbox"]
        esig = _shape_signature(entity["coords"])
        if (int(er0 + dr), int(ec0 + dc), esig) in shadow_keys:
            continue

        for rr, cc in entity["coords"]:
            nr = int(rr + dr)
            nc = int(cc + dc)
            if 0 <= nr < h and 0 <= nc < w and not entity_mask[nr, nc]:
                out[nr, nc] = SHADOW_COLOR

    return out


def _rect_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return not (
        ar + ah - 1 + pad < br
        or br + bh - 1 + pad < ar
        or ac + aw - 1 + pad < bc
        or bc + bw - 1 + pad < ac
    )


def _rect_overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    row_overlap = max(0, min(ar + ah, br + bh) - max(ar, br))
    col_overlap = max(0, min(ac + aw, bc + bw) - max(ac, bc))
    return int(row_overlap * col_overlap)


def _rect_in_bounds(rect: tuple[int, int, int, int], h: int, w: int) -> bool:
    r, c, rh, rw = rect
    return 0 <= r and 0 <= c and r + rh <= h and c + rw <= w


def _shift_rect(rect: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r, c, h, w = rect
    return r + dr, c + dc, h, w


def _paint_rect(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r, c, h, w = rect
    grid[r : r + h, c : c + w] = color


def _size_pool(dr: int, dc: int) -> list[tuple[int, int]]:
    sizes = []
    for h in range(1, SIZE_LIMIT + 1):
        for w in range(1, SIZE_LIMIT + 1):
            if h * w < 2:
                continue
            if abs(dr) >= h or abs(dc) >= w:
                sizes.append((h, w))
    return sizes


def _shuffled_positions(rng: np.random.Generator, h: int, w: int, rect_h: int, rect_w: int) -> list[tuple[int, int]]:
    positions = [(r, c) for r in range(h - rect_h + 1) for c in range(w - rect_w + 1)]
    rng.shuffle(positions)
    return positions


def _can_place(
    entity_rect: tuple[int, int, int, int],
    shadow_rect: tuple[int, int, int, int],
    entity_rects: dict[int, tuple[int, int, int, int]],
    shadow_rects: dict[int, tuple[int, int, int, int]],
    grid_h: int,
    grid_w: int,
    overlap_shadow_idx: int | None = None,
) -> bool:
    if not _rect_in_bounds(entity_rect, grid_h, grid_w) or not _rect_in_bounds(shadow_rect, grid_h, grid_w):
        return False
    if _rect_intersects(entity_rect, shadow_rect):
        return False

    for other in entity_rects.values():
        if _rect_intersects(entity_rect, other, pad=PLACEMENT_GAP):
            return False

    for idx, other in shadow_rects.items():
        if _rect_intersects(shadow_rect, other, pad=PLACEMENT_GAP):
            return False
        if _rect_intersects(entity_rect, other):
            if overlap_shadow_idx is None or idx != overlap_shadow_idx:
                return False

    for other in entity_rects.values():
        if _rect_intersects(shadow_rect, other):
            return False

    return True


def _build_grids(
    grid_h: int,
    grid_w: int,
    colors: dict[int, int],
    entity_rects: dict[int, tuple[int, int, int, int]],
    shadow_rects: dict[int, tuple[int, int, int, int]],
    paired: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    out = np.full((grid_h, grid_w), BG_COLOR, dtype=int)

    for rect in shadow_rects.values():
        _paint_rect(out, rect, SHADOW_COLOR)
    for idx in paired:
        _paint_rect(inp, shadow_rects[idx], SHADOW_COLOR)

    for idx, rect in entity_rects.items():
        color = colors[idx]
        _paint_rect(inp, rect, color)
        _paint_rect(out, rect, color)

    return inp, out


def _place_scene(
    rng: np.random.Generator,
    grid_h: int,
    grid_w: int,
    sizes: dict[int, tuple[int, int]],
    offset: tuple[int, int],
    occlusion_mode: bool,
    occluded_idx: int | None,
    blocker_idx: int | None,
) -> tuple[dict[int, tuple[int, int, int, int]], dict[int, tuple[int, int, int, int]]] | None:
    dr, dc = offset
    entity_rects: dict[int, tuple[int, int, int, int]] = {}
    shadow_rects: dict[int, tuple[int, int, int, int]] = {}

    place_order: list[int] = []
    if occlusion_mode:
        assert occluded_idx is not None and blocker_idx is not None
        place_order.extend([occluded_idx, blocker_idx])
    place_order.extend(idx for idx in sizes if idx not in place_order)

    for idx in place_order:
        rect_h, rect_w = sizes[idx]
        positions = _shuffled_positions(rng, grid_h, grid_w, rect_h, rect_w)

        placed = False
        for r, c in positions:
            entity_rect = (int(r), int(c), int(rect_h), int(rect_w))
            shadow_rect = _shift_rect(entity_rect, dr, dc)

            overlap_target = occluded_idx if occlusion_mode and idx == blocker_idx else None
            if not _can_place(
                entity_rect,
                shadow_rect,
                entity_rects,
                shadow_rects,
                grid_h,
                grid_w,
                overlap_shadow_idx=overlap_target,
            ):
                continue

            if occlusion_mode and idx == blocker_idx:
                overlap_area = _rect_overlap_area(entity_rect, shadow_rects[occluded_idx])
                if overlap_area <= 0:
                    continue
                if overlap_area >= sizes[occluded_idx][0] * sizes[occluded_idx][1]:
                    continue

            entity_rects[idx] = entity_rect
            shadow_rects[idx] = shadow_rect
            placed = True
            break

        if not placed:
            return None

    if occlusion_mode:
        assert occluded_idx is not None and blocker_idx is not None
        if _rect_overlap_area(entity_rects[blocker_idx], shadow_rects[occluded_idx]) <= 0:
            return None

    return entity_rects, shadow_rects


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    grid_h = int(rng.integers(GRID_SIDE_MIN, GRID_SIDE_MAX + 1))
    grid_w = int(rng.integers(GRID_SIDE_MIN, GRID_SIDE_MAX + 1))
    dr, dc = OFFSET_VALUES[int(rng.integers(0, len(OFFSET_VALUES)))]

    size_choices = _size_pool(dr, dc)
    n_entities = int(rng.integers(ENTITY_COUNT_MIN, ENTITY_COUNT_MAX + 1))
    if len(size_choices) < n_entities or len(ENTITY_COLORS) < n_entities:
        return None

    size_perm = rng.permutation(len(size_choices))[:n_entities]
    sizes = {idx: size_choices[int(pos)] for idx, pos in enumerate(size_perm)}

    color_perm = rng.permutation(len(ENTITY_COLORS))[:n_entities]
    colors = {idx: int(ENTITY_COLORS[int(pos)]) for idx, pos in enumerate(color_perm)}

    paired_count = int(rng.integers(2, n_entities))
    paired_order = rng.permutation(n_entities)
    paired = {int(x) for x in paired_order[:paired_count]}
    missing = [idx for idx in range(n_entities) if idx not in paired]
    assert missing

    occlusion_mode = bool(int(rng.integers(0, 2)))
    occluded_idx = None
    blocker_idx = None
    if occlusion_mode:
        occluded_idx = int(missing[int(rng.integers(0, len(missing)))])
        blockers = [idx for idx in range(n_entities) if idx != occluded_idx]
        blocker_idx = int(blockers[int(rng.integers(0, len(blockers)))])

    placed = _place_scene(rng, grid_h, grid_w, sizes, (dr, dc), occlusion_mode, occluded_idx, blocker_idx)
    if placed is None:
        return None

    entity_rects, shadow_rects = placed
    inp, out = _build_grids(grid_h, grid_w, colors, entity_rects, shadow_rects, paired)

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shadow-completion input")
