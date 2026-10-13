from __future__ import annotations

import numpy as np


_RNG = np.random.default_rng()

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# --------------- solver helpers ---------------


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return None
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox_from_mask(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in _NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not mask[nr, nc]:
                continue
            nxt = (nr, nc)
            if nxt in seen:
                continue
            seen.add(nxt)
            stack.append(nxt)
    return len(seen) == int(mask.sum())


def _is_irregular(mask: np.ndarray) -> bool:
    h, w = mask.shape
    area = int(mask.sum())
    return area >= 3 and h >= 2 and w >= 2 and area < h * w


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(mask, dtype=bool)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _label_signature(label: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(label, dtype=np.uint8)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _find_components(grid: np.ndarray, bg: int = 0) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(g[r0, c0])
            if color == bg or seen[r0, c0]:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            rows = [r for r, _c in coords]
            cols = [c for _r, c in coords]
            br0, bc0 = min(rows), min(cols)
            br1, bc1 = max(rows), max(cols)
            mask = np.zeros((br1 - br0 + 1, bc1 - bc0 + 1), dtype=bool)
            for r, c in coords:
                mask[r - br0, c - bc0] = True

            components.append(
                {
                    "color": color,
                    "coords": tuple(coords),
                    "bbox": (int(br0), int(bc0), int(br1), int(bc1)),
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )

    components.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return components


_PAIR_ASSEMBLY_CACHE: dict[tuple[tuple[int, int], bytes, tuple[int, int], bytes], list[dict]] = {}


def _pair_assemblies(mask_a: np.ndarray, mask_b: np.ndarray) -> list[dict]:
    key = _mask_signature(mask_a) + _mask_signature(mask_b)
    if key in _PAIR_ASSEMBLY_CACHE:
        return _PAIR_ASSEMBLY_CACHE[key]

    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    area_a = int(a.sum())
    area_b = int(b.sum())
    total = area_a + area_b
    ha, wa = map(int, a.shape)
    hb, wb = map(int, b.shape)

    assemblies: list[dict] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()

    for out_h in range(max(ha, hb), total + 1):
        if total % out_h != 0:
            continue
        out_w = total // out_h
        if out_w < max(wa, wb):
            continue

        for r0 in range(out_h - ha + 1):
            for c0 in range(out_w - wa + 1):
                placed_a = np.zeros((out_h, out_w), dtype=bool)
                placed_a[r0 : r0 + ha, c0 : c0 + wa] = a
                placed_b = ~placed_a
                box_b = _bbox_from_mask(placed_b)
                if box_b is None:
                    continue
                br0, bc0, br1, bc1 = box_b
                if br1 - br0 + 1 != hb or bc1 - bc0 + 1 != wb:
                    continue
                if not np.array_equal(placed_b[br0 : br1 + 1, bc0 : bc1 + 1], b):
                    continue

                label = np.where(placed_a, 1, 2).astype(np.uint8)
                sig = _label_signature(label)
                if sig in seen:
                    continue
                seen.add(sig)
                assemblies.append(
                    {
                        "label": label,
                        "height": int(out_h),
                        "width": int(out_w),
                        "a_pos": (int(r0), int(c0)),
                        "b_pos": (int(br0), int(bc0)),
                    }
                )

    assemblies.sort(
        key=lambda asm: (
            asm["height"],
            asm["width"],
            asm["a_pos"],
            asm["b_pos"],
            asm["label"].tobytes(),
        )
    )
    _PAIR_ASSEMBLY_CACHE[key] = assemblies
    return assemblies


def _search_pairings(components: list[dict], pair_options: dict[tuple[int, int], list[dict]], limit: int = 2) -> list[list[tuple[int, int, dict]]]:
    count = len(components)
    used = [False] * count
    current: list[tuple[int, int, dict]] = []
    solutions: list[list[tuple[int, int, dict]]] = []

    def rec() -> None:
        if len(solutions) >= limit:
            return

        try:
            i = next(idx for idx, flag in enumerate(used) if not flag)
        except StopIteration:
            solutions.append(list(current))
            return

        used[i] = True
        candidates: list[tuple[int, dict]] = []
        for j in range(i + 1, count):
            if used[j]:
                continue
            for asm in pair_options.get((i, j), ()):
                candidates.append((j, asm))

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1]["height"],
                item[1]["width"],
                item[1]["a_pos"],
                item[1]["b_pos"],
                item[1]["label"].tobytes(),
            )
        )

        for j, asm in candidates:
            used[j] = True
            current.append((i, j, asm))
            rec()
            current.pop()
            used[j] = False
            if len(solutions) >= limit:
                break

        used[i] = False

    rec()
    return solutions


def _build_pair_options(components: list[dict]) -> dict[tuple[int, int], list[dict]]:
    pair_options: dict[tuple[int, int], list[dict]] = {}
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            assemblies = _pair_assemblies(components[i]["mask"], components[j]["mask"])
            if assemblies:
                pair_options[(i, j)] = assemblies
    return pair_options


def _colorize_label(label: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    patch = np.empty(label.shape, dtype=int)
    patch[label == 1] = int(color_a)
    patch[label == 2] = int(color_b)
    return patch


def _pack_rectangles(rectangles: list[dict]) -> np.ndarray:
    ordered = sorted(
        rectangles,
        key=lambda rect: (
            rect["area"],
            rect["height"],
            rect["width"],
            rect["anchor_row"],
            rect["anchor_col"],
        ),
    )
    out_h = max(rect["height"] for rect in ordered)
    out_w = sum(rect["width"] for rect in ordered) + len(ordered) - 1
    out = np.zeros((out_h, out_w), dtype=int)
    col = 0
    for rect in ordered:
        patch = rect["patch"]
        h, w = patch.shape
        out[out_h - h : out_h, col : col + w] = patch
        col += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    components = _find_components(g, bg=0)
    assert 4 <= len(components) <= 8 and len(components) % 2 == 0

    pair_options = _build_pair_options(components)
    solutions = _search_pairings(components, pair_options, limit=2)
    assert len(solutions) == 1, "expected a unique pairing solution"

    rectangles: list[dict] = []
    for i, j, asm in solutions[0]:
        comp_a = components[i]
        comp_b = components[j]
        rectangles.append(
            {
                "patch": _colorize_label(asm["label"], comp_a["color"], comp_b["color"]),
                "area": int(asm["height"] * asm["width"]),
                "height": int(asm["height"]),
                "width": int(asm["width"]),
                "anchor_row": min(comp_a["bbox"][0], comp_b["bbox"][0]),
                "anchor_col": min(comp_a["bbox"][1], comp_b["bbox"][1]),
            }
        )

    return _pack_rectangles(rectangles)


# --------------- generator helpers ---------------


PAIR_COUNT_VALUES = (2, 3, 4)
RECT_DIMS = tuple((h, w) for h in range(3, 7) for w in range(3, 7))
GRID_MIN_SIDE = 18
GRID_MAX_SIDE = 30
MAX_RETRIES = 400
PAIR_BUILD_RETRIES = 200
PLACEMENT_RETRIES = 60


def _grow_connected_subset(h: int, w: int, target: int) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    sr = int(_RNG.integers(0, h))
    sc = int(_RNG.integers(0, w))
    mask[sr, sc] = True

    while int(mask.sum()) < target:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            for dr, dc in _NEIGHBORS4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        nr, nc = tuple(frontier)[int(_RNG.integers(0, len(frontier)))]
        mask[nr, nc] = True
    return mask


def _random_transform(label: np.ndarray) -> np.ndarray:
    out = np.rot90(label, int(_RNG.integers(0, 4)))
    if bool(_RNG.integers(0, 2)):
        out = np.fliplr(out)
    return np.asarray(out, dtype=np.uint8)


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _build_pair(rect_h: int, rect_w: int) -> dict | None:
    area = rect_h * rect_w
    min_a = max(3, area // 3)
    max_a = min(area - 3, (2 * area) // 3)
    if min_a > max_a:
        return None

    for _ in range(PAIR_BUILD_RETRIES):
        target_a = int(_RNG.integers(min_a, max_a + 1))
        mask_a_full = _grow_connected_subset(rect_h, rect_w, target_a)
        if mask_a_full is None:
            continue
        mask_b_full = ~mask_a_full
        if not _is_connected(mask_b_full):
            continue

        label = np.where(mask_a_full, 1, 2).astype(np.uint8)
        label = _random_transform(label)
        mask_a = _trim_mask(label == 1)
        mask_b = _trim_mask(label == 2)
        if not (_is_irregular(mask_a) and _is_irregular(mask_b)):
            continue

        assemblies = _pair_assemblies(mask_a, mask_b)
        if len(assemblies) != 1:
            continue

        return {
            "label": assemblies[0]["label"],
            "height": int(assemblies[0]["height"]),
            "width": int(assemblies[0]["width"]),
            "area": int(assemblies[0]["height"] * assemblies[0]["width"]),
            "mask_a": mask_a,
            "mask_b": mask_b,
        }
    return None


def _dilate8(mask: np.ndarray) -> np.ndarray:
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


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != 0
    sub = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    sub[mask] = patch[mask]
    grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = sub


def _place_objects(objects: list[dict], grid_h: int, grid_w: int) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(objects)
    order = sorted(
        range(len(objects)),
        key=lambda idx: (objects[idx]["patch"].shape[0] * objects[idx]["patch"].shape[1], objects[idx]["area"]),
        reverse=True,
    )

    for idx in order:
        patch = objects[idx]["patch"]
        mask = objects[idx]["mask"]
        h, w = patch.shape
        if h > grid_h or w > grid_w:
            return None

        dilated = _dilate8(occupied)
        choices: list[tuple[int, int]] = []
        for r0 in range(grid_h - h + 1):
            window = dilated[r0 : r0 + h, :]
            for c0 in range(grid_w - w + 1):
                if not np.any(window[:, c0 : c0 + w] & mask):
                    choices.append((int(r0), int(c0)))
        if not choices:
            return None

        r0, c0 = choices[int(_RNG.integers(0, len(choices)))]
        placements[idx] = (r0, c0)
        _paint_patch(grid, patch, r0, c0)
        occupied[r0 : r0 + h, c0 : c0 + w] |= mask

    return grid, [pos for pos in placements if pos is not None]


def _expected_output(pairs: list[dict]) -> np.ndarray:
    rectangles = []
    for idx, pair in enumerate(sorted(pairs, key=lambda item: item["area"])):
        rectangles.append(
            {
                "patch": pair["rect_patch"],
                "area": int(pair["area"]),
                "height": int(pair["height"]),
                "width": int(pair["width"]),
                "anchor_row": idx,
                "anchor_col": 0,
            }
        )
    return _pack_rectangles(rectangles)


def _choose_rectangles(pair_count: int) -> list[tuple[int, int]] | None:
    area_to_dims: dict[int, list[tuple[int, int]]] = {}
    for dims in RECT_DIMS:
        area_to_dims.setdefault(dims[0] * dims[1], []).append(dims)
    areas = list(area_to_dims)
    _RNG.shuffle(areas)
    chosen_areas = areas[:pair_count]
    if len(chosen_areas) != pair_count:
        return None
    return [area_to_dims[area][int(_RNG.integers(0, len(area_to_dims[area])))] for area in chosen_areas]


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(_RNG.choice(PAIR_COUNT_VALUES))
    rect_dims = _choose_rectangles(pair_count)
    if rect_dims is None:
        return None

    pairs_raw: list[dict] = []
    for rect_h, rect_w in rect_dims:
        pair = _build_pair(rect_h, rect_w)
        if pair is None:
            return None
        pairs_raw.append(pair)

    colors = list(_RNG.choice(np.arange(1, 10), size=2 * pair_count, replace=False))
    _RNG.shuffle(colors)

    pairs: list[dict] = []
    objects: list[dict] = []
    color_idx = 0
    for pair in pairs_raw:
        color_a = int(colors[color_idx])
        color_b = int(colors[color_idx + 1])
        color_idx += 2

        rect_patch = _colorize_label(pair["label"], color_a, color_b)
        patch_a = _mask_to_patch(pair["mask_a"], color_a)
        patch_b = _mask_to_patch(pair["mask_b"], color_b)
        pairs.append(
            {
                "rect_patch": rect_patch,
                "area": int(pair["area"]),
                "height": int(pair["height"]),
                "width": int(pair["width"]),
            }
        )
        objects.append({"patch": patch_a, "mask": pair["mask_a"], "area": int(pair["mask_a"].sum())})
        objects.append({"patch": patch_b, "mask": pair["mask_b"], "area": int(pair["mask_b"].sum())})

    expected = _expected_output(pairs)

    for _ in range(PLACEMENT_RETRIES):
        grid_h = int(_RNG.integers(GRID_MIN_SIDE, GRID_MAX_SIDE + 1))
        grid_w = int(_RNG.integers(GRID_MIN_SIDE, GRID_MAX_SIDE + 1))
        placed = _place_objects(objects, grid_h, grid_w)
        if placed is None:
            continue
        inp, _placements = placed

        components = _find_components(inp, bg=0)
        if len(components) != 2 * pair_count:
            continue
        if len({comp["color"] for comp in components}) != 2 * pair_count:
            continue
        if not all(_is_irregular(comp["mask"]) for comp in components):
            continue

        pair_options = _build_pair_options(components)
        solutions = _search_pairings(components, pair_options, limit=2)
        if len(solutions) != 1:
            continue

        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
