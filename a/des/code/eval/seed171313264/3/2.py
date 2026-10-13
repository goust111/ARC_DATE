import numpy as np


_RNG = np.random.default_rng()
_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator configuration.
_GRID_MIN = 18
_GRID_MAX = 30
_PAIR_COUNT_CHOICES = (2, 3, 4)
_RECT_H_RANGE = (3, 7)  # inclusive lower, exclusive upper
_RECT_W_RANGE = (3, 8)
_MIN_FRAGMENT_AREA = 3
_PAIR_RETRIES = 80
_PLACEMENT_RETRIES = 250
_GENERATE_RETRIES = 300
_COLOR_POOL = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].copy()


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == 0 or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            pts = np.array(coords, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True

            components.append(
                {
                    "color": color,
                    "area": int(mask.sum()),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                }
            )

    return components


def _pair_candidates(mask_a: np.ndarray, mask_b: np.ndarray) -> list[dict]:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    ha, wa = a.shape
    hb, wb = b.shape
    total_area = int(a.sum() + b.sum())

    candidates: list[dict] = []
    seen = set()
    for dr in range(-hb, ha + 1):
        for dc in range(-wb, wa + 1):
            top = min(0, dr)
            left = min(0, dc)
            bottom = max(ha, dr + hb)
            right = max(wa, dc + wb)
            out_h = int(bottom - top)
            out_w = int(right - left)
            if out_h * out_w != total_area:
                continue

            a_r, a_c = int(-top), int(-left)
            b_r, b_c = int(dr - top), int(dc - left)

            occ = np.zeros((out_h, out_w), dtype=np.uint8)
            occ[a_r : a_r + ha, a_c : a_c + wa][a] = 1
            if np.any(occ[b_r : b_r + hb, b_c : b_c + wb][b]):
                continue
            occ[b_r : b_r + hb, b_c : b_c + wb][b] = 2

            key = (out_h, out_w, a_r, a_c, b_r, b_c)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "shape": (out_h, out_w),
                    "a_pos": (a_r, a_c),
                    "b_pos": (b_r, b_c),
                    "area": int(total_area),
                }
            )

    candidates.sort(key=lambda cand: (cand["shape"][0], cand["shape"][1], cand["a_pos"], cand["b_pos"]))
    return candidates


def _render_candidate(candidate: dict, comp_a: dict, comp_b: dict) -> np.ndarray:
    out_h, out_w = candidate["shape"]
    out = np.zeros((out_h, out_w), dtype=int)

    a_r, a_c = candidate["a_pos"]
    b_r, b_c = candidate["b_pos"]
    a_mask = comp_a["mask"]
    b_mask = comp_b["mask"]
    out[a_r : a_r + a_mask.shape[0], a_c : a_c + a_mask.shape[1]][a_mask] = int(comp_a["color"])
    out[b_r : b_r + b_mask.shape[0], b_c : b_c + b_mask.shape[1]][b_mask] = int(comp_b["color"])
    return out


def _pair_graph(components: list[dict]) -> tuple[dict, dict]:
    compat: dict[tuple[int, int], dict] = {}
    counts: dict[tuple[int, int], int] = {}

    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            if int(components[i]["color"]) == int(components[j]["color"]):
                continue
            candidates = _pair_candidates(components[i]["mask"], components[j]["mask"])
            if not candidates:
                continue
            compat[(i, j)] = candidates[0]
            counts[(i, j)] = len(candidates)

    return compat, counts


def _enumerate_matchings(
    remaining: tuple[int, ...],
    compat: dict[tuple[int, int], dict],
    current: list[tuple[int, int]],
    solutions: list[list[tuple[int, int]]],
    limit: int,
) -> None:
    if len(solutions) >= limit:
        return
    if not remaining:
        solutions.append(list(current))
        return

    i = remaining[0]
    for idx in range(1, len(remaining)):
        j = remaining[idx]
        if (i, j) not in compat:
            continue
        current.append((i, j))
        next_remaining = remaining[1:idx] + remaining[idx + 1 :]
        _enumerate_matchings(next_remaining, compat, current, solutions, limit)
        current.pop()
        if len(solutions) >= limit:
            return


def _solve_components(components: list[dict], limit: int = 2) -> tuple[list[list[tuple[int, int, dict]]], dict, dict]:
    if not components or len(components) % 2 != 0:
        return [], {}, {}

    compat, counts = _pair_graph(components)
    raw_solutions: list[list[tuple[int, int]]] = []
    _enumerate_matchings(tuple(range(len(components))), compat, [], raw_solutions, limit)

    solutions: list[list[tuple[int, int, dict]]] = []
    for raw in raw_solutions:
        detailed = [(i, j, compat[(i, j)]) for i, j in raw]
        detailed.sort(key=lambda item: (item[0], item[1]))
        solutions.append(detailed)
    solutions.sort(
        key=lambda sol: tuple(
            (i, j, cand["shape"][0], cand["shape"][1], cand["a_pos"], cand["b_pos"]) for i, j, cand in sol
        )
    )
    return solutions, compat, counts


def _stack_rectangles(rectangles: list[np.ndarray]) -> np.ndarray:
    total_h = sum(rect.shape[0] for rect in rectangles) + len(rectangles) - 1
    total_w = max(rect.shape[1] for rect in rectangles)
    out = np.zeros((total_h, total_w), dtype=int)

    r = 0
    for idx, rect in enumerate(rectangles):
        h, w = rect.shape
        out[r : r + h, :w] = rect
        r += h + (0 if idx == len(rectangles) - 1 else 1)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair monochrome fragments that translate together into solid rectangles,
    then stack the assembled rectangles from smallest area to largest.
    """
    g = np.asarray(grid, dtype=int)
    components = _find_components(g)
    assert components and len(components) % 2 == 0

    solutions, _compat, _counts = _solve_components(components, limit=2)
    assert solutions, "expected at least one perfect complementary pairing"
    solution = solutions[0]

    rectangles: list[tuple[int, int, int, np.ndarray]] = []
    for i, j, candidate in solution:
        patch = _render_candidate(candidate, components[i], components[j])
        area = int(candidate["area"])
        rectangles.append((area, patch.shape[0], patch.shape[1], patch))

    rectangles.sort(key=lambda item: (item[0], item[1], item[2]))
    return _stack_rectangles([patch for _area, _h, _w, patch in rectangles])


def _make_cut_pair(rect_h: int, rect_w: int, orientation: str) -> dict | None:
    for _ in range(_PAIR_RETRIES):
        if orientation == "vertical":
            cuts = _RNG.integers(1, rect_w, size=rect_h)
            if rect_h >= 2 and np.all(cuts == cuts[0]) and _RNG.random() < 0.75:
                continue
            mask_a = np.zeros((rect_h, rect_w), dtype=bool)
            for r, cut in enumerate(cuts):
                mask_a[r, : int(cut)] = True
        else:
            cuts = _RNG.integers(1, rect_h, size=rect_w)
            if rect_w >= 2 and np.all(cuts == cuts[0]) and _RNG.random() < 0.75:
                continue
            mask_a = np.zeros((rect_h, rect_w), dtype=bool)
            for c, cut in enumerate(cuts):
                mask_a[: int(cut), c] = True

        if _RNG.random() < 0.5:
            mask_a = ~mask_a
        mask_b = ~mask_a

        if min(int(mask_a.sum()), int(mask_b.sum())) < _MIN_FRAGMENT_AREA:
            continue

        cropped_a = _tight_crop(mask_a)
        cropped_b = _tight_crop(mask_b)
        candidates = _pair_candidates(cropped_a, cropped_b)
        if len(candidates) != 1:
            continue

        return {
            "area": int(rect_h * rect_w),
            "orientation": orientation,
            "comp_a_mask": cropped_a,
            "comp_b_mask": cropped_b,
            "candidate": candidates[0],
        }

    return None


def _make_component_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _place_patches(patches: list[np.ndarray], grid_h: int, grid_w: int) -> np.ndarray | None:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    for idx in _RNG.permutation(len(patches)):
        patch = patches[int(idx)]
        h, w = patch.shape
        if h + 2 > grid_h or w + 2 > grid_w:
            return None

        placed = False
        row_hi = grid_h - h + 1
        col_hi = grid_w - w + 1
        for _ in range(_PLACEMENT_RETRIES):
            r0 = int(_RNG.integers(0, row_hi))
            c0 = int(_RNG.integers(0, col_hi))
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + h + 1)
            cc1 = min(grid_w, c0 + w + 1)
            if np.any(blocked[rr0:rr1, cc0:cc1]):
                continue

            mask = patch != 0
            grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
            blocked[rr0:rr1, cc0:cc1] = True
            placed = True
            break

        if not placed:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    num_pairs = int(_RNG.choice(_PAIR_COUNT_CHOICES))

    pair_specs: list[dict] = []
    used_areas = set()
    for _ in range(num_pairs):
        built = None
        for _ in range(_PAIR_RETRIES):
            rect_h = int(_RNG.integers(*_RECT_H_RANGE))
            rect_w = int(_RNG.integers(*_RECT_W_RANGE))
            area = int(rect_h * rect_w)
            if area in used_areas:
                continue
            orientation = "vertical" if _RNG.random() < 0.5 else "horizontal"
            built = _make_cut_pair(rect_h, rect_w, orientation)
            if built is None:
                continue
            used_areas.add(area)
            pair_specs.append(built)
            break
        if built is None:
            return None

    palette_size = min(5, num_pairs + 1)
    palette = _RNG.choice(_COLOR_POOL, size=palette_size, replace=False)

    components: list[dict] = []
    expected_rectangles: list[tuple[int, np.ndarray]] = []
    for pair_id, spec in enumerate(pair_specs):
        colors = _RNG.choice(palette, size=2, replace=False)
        comp_a = {"mask": spec["comp_a_mask"], "color": int(colors[0]), "pair_id": pair_id}
        comp_b = {"mask": spec["comp_b_mask"], "color": int(colors[1]), "pair_id": pair_id}
        components.extend([comp_a, comp_b])
        expected_rectangles.append((int(spec["area"]), _render_candidate(spec["candidate"], comp_a, comp_b)))

    compat, counts = _pair_graph(components)
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            key = (i, j)
            same_pair = int(components[i]["pair_id"]) == int(components[j]["pair_id"])
            if same_pair:
                if counts.get(key) != 1:
                    return None
            elif key in compat:
                return None

    ordered_rectangles = [rect for _area, rect in sorted(expected_rectangles, key=lambda item: (item[0], item[1].shape))]
    expected = _stack_rectangles(ordered_rectangles)
    if expected.shape[0] > _GRID_MAX or expected.shape[1] > _GRID_MAX:
        return None

    patches = [_make_component_patch(comp["mask"], int(comp["color"])) for comp in components]
    total_bbox_area = sum(int(patch.shape[0] * patch.shape[1]) for patch in patches)
    min_side = max(
        _GRID_MIN,
        max(patch.shape[0] for patch in patches) + 2,
        max(patch.shape[1] for patch in patches) + 2,
        int(np.ceil(np.sqrt(total_bbox_area * 2.2))),
    )
    if min_side > _GRID_MAX:
        return None

    grid_h = int(_RNG.integers(min_side, _GRID_MAX + 1))
    grid_w = int(_RNG.integers(min_side, _GRID_MAX + 1))
    grid = _place_patches(patches, grid_h, grid_w)
    if grid is None:
        return None

    found_components = _find_components(grid)
    if len(found_components) != len(components):
        return None
    solutions, _compat2, _counts2 = _solve_components(found_components, limit=3)
    if len(solutions) != 1:
        return None
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragments sample")
