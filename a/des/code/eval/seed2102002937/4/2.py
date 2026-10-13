import numpy as np
from collections import Counter, deque


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG = 8
GRID_MAX = 30
GEN_RETRIES = 300
_RNG = np.random.default_rng()


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _connected_components_for_color(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == color
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            cropped = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            cropped[arr[:, 0] - r0, arr[:, 1] - c0] = True
            comps.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": cropped,
                }
            )
    return comps


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)


def _pair_center(comp_a: dict, comp_b: dict) -> tuple[float, float]:
    ra, ca = _bbox_center(comp_a["bbox"])
    rb, cb = _bbox_center(comp_b["bbox"])
    return ((ra + rb) / 2.0, (ca + cb) / 2.0)


def _cover_candidate_vertical(mask_a: np.ndarray, mask_b: np.ndarray) -> list[tuple[str, int, int, int]]:
    if mask_a.shape[1] != mask_b.shape[1]:
        return []

    h1, w = mask_a.shape
    h2 = mask_b.shape[0]
    flipped = np.fliplr(mask_b)
    out: list[tuple[str, int, int, int]] = []

    for shift in range(-h2, h1 + 1):
        top = min(0, shift)
        bottom = max(h1, shift + h2)
        canvas_a = np.zeros((bottom - top, w), dtype=bool)
        canvas_b = np.zeros_like(canvas_a)
        canvas_a[-top : -top + h1, :] = mask_a
        canvas_b[shift - top : shift - top + h2, :] = flipped
        if not np.any(canvas_a & canvas_b) and np.all(canvas_a | canvas_b):
            out.append(("vertical", int(bottom - top), int(w), int(shift)))
    return out


def _cover_candidate_horizontal(mask_a: np.ndarray, mask_b: np.ndarray) -> list[tuple[str, int, int, int]]:
    if mask_a.shape[0] != mask_b.shape[0]:
        return []

    h, w1 = mask_a.shape
    w2 = mask_b.shape[1]
    flipped = np.flipud(mask_b)
    out: list[tuple[str, int, int, int]] = []

    for shift in range(-w2, w1 + 1):
        left = min(0, shift)
        right = max(w1, shift + w2)
        canvas_a = np.zeros((h, right - left), dtype=bool)
        canvas_b = np.zeros_like(canvas_a)
        canvas_a[:, -left : -left + w1] = mask_a
        canvas_b[:, shift - left : shift - left + w2] = flipped
        if not np.any(canvas_a & canvas_b) and np.all(canvas_a | canvas_b):
            out.append(("horizontal", int(h), int(right - left), int(shift)))
    return out


def _merge_candidates(mask_a: np.ndarray, mask_b: np.ndarray) -> list[tuple[str, int, int, int]]:
    return _cover_candidate_vertical(mask_a, mask_b) + _cover_candidate_horizontal(mask_a, mask_b)


def _pack_rectangles(rects: list[dict], bg: int) -> np.ndarray:
    assert rects
    out_h = max(rect["shape"][0] for rect in rects)
    out_w = sum(rect["shape"][1] for rect in rects) + len(rects) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    c = 0
    for rect in rects:
        h, w = rect["shape"]
        out[:h, c : c + w] = int(rect["color"])
        c += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors

    merged: list[dict] = []
    for color in colors:
        comps = _connected_components_for_color(g, color)
        assert len(comps) == 2
        candidates = _merge_candidates(comps[0]["mask"], comps[1]["mask"])
        assert len(candidates) == 1
        _orientation, out_h, out_w, _shift = candidates[0]
        merged.append(
            {
                "color": color,
                "shape": (int(out_h), int(out_w)),
                "center": _pair_center(comps[0], comps[1]),
            }
        )

    merged.sort(key=lambda item: (item["center"][0], item["center"][1]))
    return _pack_rectangles(merged, bg)


def _build_row_prefix_mask(width: int, lengths: list[int]) -> np.ndarray:
    mask = np.zeros((len(lengths), width), dtype=bool)
    for r, length in enumerate(lengths):
        if length > 0:
            mask[r, :length] = True
    return mask


def _joint_random_transforms(mask_a: np.ndarray, mask_b: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    a = mask_a.copy()
    b = mask_b.copy()
    if int(rng.integers(0, 2)):
        a = np.flipud(a)
        b = np.flipud(b)
    if int(rng.integers(0, 2)):
        a = np.fliplr(a)
        b = np.fliplr(b)
    if int(rng.integers(0, 2)):
        a, b = b, a
    return a.copy(), b.copy()


def _build_base_vertical_pair(merged_h: int, merged_w: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    middle = merged_h - 2
    assert middle >= 2
    if merged_w < 3:
        return None

    for _ in range(80):
        seq = rng.integers(1, merged_w, size=middle).tolist()
        if len(set(seq)) < 2:
            continue

        lengths_a = [merged_w] + seq
        lengths_b = [merged_w - value for value in seq] + [merged_w]
        mask_a = _build_row_prefix_mask(merged_w, lengths_a)
        mask_b = _build_row_prefix_mask(merged_w, lengths_b)
        mask_a, mask_b = _joint_random_transforms(mask_a, mask_b, rng)

        candidates = _merge_candidates(mask_a, mask_b)
        if len(candidates) != 1:
            continue
        orient, out_h, out_w, _shift = candidates[0]
        if orient == "vertical" and out_h == merged_h and out_w == merged_w:
            return mask_a, mask_b
    return None


def _build_pair(merged_h: int, merged_w: int, orientation: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    if orientation == "vertical":
        return _build_base_vertical_pair(merged_h, merged_w, rng)

    if orientation == "horizontal":
        base = _build_base_vertical_pair(merged_w, merged_h, rng)
        if base is None:
            return None
        return base[0].T.copy(), base[1].T.copy()

    raise ValueError(f"unknown orientation: {orientation}")


def _expanded_slices(r0: int, c0: int, h: int, w: int, grid_h: int, grid_w: int) -> tuple[slice, slice]:
    return (
        slice(max(0, r0 - 1), min(grid_h, r0 + h + 1)),
        slice(max(0, c0 - 1), min(grid_w, c0 + w + 1)),
    )


def _place_fragments(grid_h: int, grid_w: int, fragments: list[dict], rng: np.random.Generator) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[dict | None] = [None] * len(fragments)

    order = sorted(range(len(fragments)), key=lambda idx: int(fragments[idx]["mask"].sum()), reverse=True)
    for idx in order:
        mask = fragments[idx]["mask"]
        color = int(fragments[idx]["color"])
        h, w = mask.shape
        placed = False

        for _ in range(200):
            r0 = int(rng.integers(0, grid_h - h + 1))
            c0 = int(rng.integers(0, grid_w - w + 1))
            rs, cs = _expanded_slices(r0, c0, h, w, grid_h, grid_w)
            if np.any(blocked[rs, cs]):
                continue

            patch = grid[r0 : r0 + h, c0 : c0 + w]
            if np.any(patch[mask] != BG):
                continue

            patch[mask] = color
            blocked[rs, cs] = True
            placements[idx] = {
                "pair_id": int(fragments[idx]["pair_id"]),
                "color": color,
                "bbox": (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)),
            }
            placed = True
            break

        if not placed:
            return None

    return grid, [item for item in placements if item is not None]


def _orientation_schedule(count: int, rng: np.random.Generator) -> list[str]:
    if count == 1:
        return [str(rng.choice(["vertical", "horizontal"]))]
    values = ["vertical", "horizontal"]
    while len(values) < count:
        values.append(str(rng.choice(["vertical", "horizontal"])))
    rng.shuffle(values)
    return values


def _expected_output_from_generation(group_info: list[dict]) -> np.ndarray:
    rects = []
    for item in group_info:
        rects.append(
            {
                "color": int(item["color"]),
                "shape": tuple(item["shape"]),
                "center": tuple(item["center"]),
            }
        )
    rects.sort(key=lambda item: (item["center"][0], item["center"][1]))
    return _pack_rectangles(rects, BG)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    group_count = int(rng.integers(2, 5))
    palette = [color for color in range(10) if color != BG]
    rng.shuffle(palette)
    colors = palette[:group_count]
    orientations = _orientation_schedule(group_count, rng)

    merged_shapes: list[tuple[int, int]] = []
    while True:
        merged_shapes = []
        for _ in range(group_count):
            merged_h = int(rng.integers(4, 8))
            merged_w = int(rng.integers(4, 8))
            merged_shapes.append((merged_h, merged_w))
        if sum(shape[1] for shape in merged_shapes) + group_count - 1 <= GRID_MAX:
            break

    fragments: list[dict] = []
    pair_shapes: dict[int, tuple[int, int]] = {}
    for pair_id, (color, orientation, shape) in enumerate(zip(colors, orientations, merged_shapes)):
        merged_h, merged_w = shape
        pair = _build_pair(merged_h, merged_w, orientation, rng)
        if pair is None:
            return None
        pair_shapes[pair_id] = (merged_h, merged_w)
        fragments.append({"pair_id": pair_id, "color": color, "mask": pair[0]})
        fragments.append({"pair_id": pair_id, "color": color, "mask": pair[1]})

    side_floor = max(18, 12 + 3 * group_count, max(max(mask.shape) for mask in (frag["mask"] for frag in fragments)) + 4)
    grid_h = int(rng.integers(side_floor, GRID_MAX + 1))
    grid_w = int(rng.integers(side_floor, GRID_MAX + 1))
    placed = _place_fragments(grid_h, grid_w, fragments, rng)
    if placed is None:
        return None
    grid, placements = placed

    by_pair: dict[int, list[dict]] = {}
    for item in placements:
        by_pair.setdefault(int(item["pair_id"]), []).append(item)

    if set(by_pair) != set(range(group_count)):
        return None

    group_info: list[dict] = []
    for pair_id in range(group_count):
        parts = by_pair[pair_id]
        if len(parts) != 2:
            return None
        fake_a = {"bbox": parts[0]["bbox"]}
        fake_b = {"bbox": parts[1]["bbox"]}
        group_info.append(
            {
                "color": int(parts[0]["color"]),
                "shape": pair_shapes[pair_id],
                "center": _pair_center(fake_a, fake_b),
            }
        )

    expected = _expected_output_from_generation(group_info)
    if expected.shape[0] > GRID_MAX or expected.shape[1] > GRID_MAX:
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
