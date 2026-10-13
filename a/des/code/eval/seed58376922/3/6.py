import random
from collections import deque

import numpy as np


BG = 0
RED = 2
GREY = 5
SHAPE_COLORS = (1, 3, 4, 6, 7, 8, 9)
OPS = ("lr", "ud", "cw", "ccw")
MAX_GRID_SIZE = 30
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _component_patch(grid: np.ndarray, coords: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    return patch, (r0, c0, r1, c1)


def _is_rect_frame(grid: np.ndarray, color: int, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return (
        int(coords.shape[0]) == int(border.sum())
        and np.all(sub[border] == color)
        and np.all(sub[~border] != color)
    )


def _find_lookup_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates = []
    for color in [int(v) for v in np.unique(grid) if int(v) != bg]:
        for comp in _components(grid == color):
            if not _is_rect_frame(grid, color, comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0 or not np.any(inner == RED):
                continue
            candidates.append((int(color), (r0, c0, r1, c1)))
    assert len(candidates) == 1, f"expected one lookup frame, found {len(candidates)}"
    return candidates[0]


def _classify_operator(patch: np.ndarray) -> str:
    mask = patch == RED
    h, w = mask.shape
    area = int(mask.sum())
    if area == 2 and h == 2 and w == 1:
        return "lr"
    if area == 2 and h == 1 and w == 2:
        return "ud"
    if area == 3 and h == 2 and w == 2:
        if not mask[1, 0] and mask[1, 1]:
            return "cw"
        if mask[1, 0] and not mask[1, 1]:
            return "ccw"
    raise AssertionError(f"unrecognized operator patch:\n{patch}")


def _extract_lookup_entries(
    grid: np.ndarray,
    frame_color: int,
    frame_bbox: tuple[int, int, int, int],
    bg: int,
) -> list[dict]:
    r0, c0, r1, c1 = frame_bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]

    key_mask = (inner != bg) & (inner != frame_color) & (inner != RED)
    key_comps = _components(key_mask)
    op_comps = _components(inner == RED)
    assert key_comps and len(key_comps) == len(op_comps)

    keys = []
    for comp in key_comps:
        patch, bbox = _component_patch(inner, comp, bg)
        keys.append({"bbox": bbox, "patch": patch})

    ops = []
    for comp in op_comps:
        patch, bbox = _component_patch(inner, comp, bg)
        ops.append({"bbox": bbox, "op": _classify_operator(patch)})

    assignments: dict[int, str] = {}
    used_keys: set[int] = set()
    for op in ops:
        or0, oc0, or1, _ = op["bbox"]
        best = None
        for idx, key in enumerate(keys):
            kr0, kc0, kr1, kc1 = key["bbox"]
            overlap = min(or1, kr1) - max(or0, kr0) + 1
            gap = oc0 - kc1 - 1
            if overlap <= 0 or gap < 0:
                continue
            score = (gap, abs(or0 - kr0), idx)
            if best is None or score < best[0]:
                best = (score, idx)
        assert best is not None, "every operator must have a key immediately to its left"
        idx = best[1]
        assert idx not in assignments, "each key must map to exactly one operator"
        assignments[idx] = op["op"]
        used_keys.add(idx)

    assert len(assignments) == len(keys) == len(used_keys)
    return [{"patch": keys[idx]["patch"], "op": assignments[idx]} for idx in range(len(keys))]


def _scale_patch(patch: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(patch, np.ones((scale, scale), dtype=int))


def _apply_op(patch: np.ndarray, op: str) -> np.ndarray:
    if op == "lr":
        return np.fliplr(patch)
    if op == "ud":
        return np.flipud(patch)
    if op == "cw":
        return np.rot90(patch, k=3)
    if op == "ccw":
        return np.rot90(patch, k=1)
    raise AssertionError(f"unknown op {op}")


def _match_entry(patch: np.ndarray, entries: list[dict]) -> dict:
    h, w = patch.shape
    matches = []
    for entry in entries:
        kh, kw = entry["patch"].shape
        if h % kh or w % kw:
            continue
        scale_h = h // kh
        scale_w = w // kw
        if scale_h != scale_w or scale_h < 1:
            continue
        if np.array_equal(_scale_patch(entry["patch"], scale_h), patch):
            matches.append(entry)
    assert len(matches) == 1, f"expected one matching entry, found {len(matches)}"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each large colored shape to a lookup-table key by its downscaled pattern,
    read the adjacent red operator, apply the corresponding geometric transform,
    and write the transformed shape back in place.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frame_color, frame_bbox = _find_lookup_frame(g, bg)
    entries = _extract_lookup_entries(g, frame_color, frame_bbox, bg)

    out = g.copy()
    outside = g != bg
    r0, c0, r1, c1 = frame_bbox
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    shape_comps = _components(outside)
    assert shape_comps, "expected shapes outside the lookup frame"

    for comp in shape_comps:
        patch, (sr0, sc0, sr1, sc1) = _component_patch(g, comp, bg)
        entry = _match_entry(patch, entries)
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = _apply_op(patch, entry["op"])
    return out


def _mask_connected(mask: np.ndarray) -> bool:
    return len(_components(mask.astype(bool))) == 1


def _all_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    cur = mask.copy()
    for _ in range(4):
        variants.append(cur.copy())
        cur = np.rot90(cur)
    cur = np.fliplr(mask)
    for _ in range(4):
        variants.append(cur.copy())
        cur = np.rot90(cur)
    return variants


def _canonical_mask(mask: np.ndarray) -> bytes:
    return min(variant.astype(np.uint8).tobytes() for variant in _all_variants(mask))


def _is_good_mask(mask: np.ndarray) -> bool:
    if not mask[0, :].any() or not mask[-1, :].any():
        return False
    if not mask[:, 0].any() or not mask[:, -1].any():
        return False
    if not _mask_connected(mask):
        return False
    if mask.all():
        return False
    reps = {variant.astype(np.uint8).tobytes() for variant in [mask, np.fliplr(mask), np.flipud(mask), np.rot90(mask, 1), np.rot90(mask, 3)]}
    return len(reps) == 5


def _random_mask(size: int) -> np.ndarray:
    for _ in range(300):
        target = random.randint(size + 1, size * size - 2)
        mask = np.zeros((size, size), dtype=bool)
        start = (random.randrange(size), random.randrange(size))
        mask[start] = True
        while int(mask.sum()) < target:
            frontier = []
            seen = set()
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < size and 0 <= nc < size and not mask[nr, nc] and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(frontier)
            mask[nr, nc] = True
        if _is_good_mask(mask):
            return mask
    raise RuntimeError("failed to generate an asymmetric mask")


def _symbol_patch(op: str) -> np.ndarray:
    if op == "lr":
        return np.array([[RED], [RED]], dtype=int)
    if op == "ud":
        return np.array([[RED, RED]], dtype=int)
    if op == "cw":
        return np.array([[RED, RED], [BG, RED]], dtype=int)
    if op == "ccw":
        return np.array([[RED, RED], [RED, BG]], dtype=int)
    raise AssertionError(f"unknown op {op}")


def _build_lookup_entries() -> list[dict]:
    colors = list(SHAPE_COLORS)
    random.shuffle(colors)
    ops = list(OPS)
    random.shuffle(ops)
    used = set()
    entries = []
    for op, color in zip(ops, colors, strict=False):
        for _ in range(200):
            size = random.choice((3, 4))
            mask = _random_mask(size)
            canon = _canonical_mask(mask)
            if canon in used:
                continue
            used.add(canon)
            patch = np.where(mask, color, BG).astype(int)
            entries.append({"op": op, "patch": patch, "size": size})
            break
        else:
            raise RuntimeError("failed to sample unique lookup keys")
    return entries


def _build_lookup_patch(entries: list[dict]) -> np.ndarray:
    gaps = [random.randint(1, 2) for _ in range(len(entries) - 1)]
    widths = [entry["size"] + _symbol_patch(entry["op"]).shape[1] for entry in entries]
    content_w = sum(widths) + sum(gaps)
    lookup = np.full((6, content_w + 2), BG, dtype=int)
    lookup[0, :] = GREY
    lookup[-1, :] = GREY
    lookup[:, 0] = GREY
    lookup[:, -1] = GREY

    cursor = 1
    for idx, entry in enumerate(entries):
        patch = entry["patch"]
        size = entry["size"]
        top = 1 if size == 4 else random.choice((1, 2))
        left = cursor
        key_view = lookup[top : top + size, left : left + size]
        key_view[patch != BG] = patch[patch != BG]

        symbol = _symbol_patch(entry["op"])
        sym_view = lookup[top : top + symbol.shape[0], left + size : left + size + symbol.shape[1]]
        sym_view[symbol != BG] = RED

        cursor += widths[idx]
        if idx < len(gaps):
            cursor += gaps[idx]
    return lookup


def _partition_counts(n: int) -> list[int]:
    options = {
        4: ([2, 2],),
        5: ([2, 3], [3, 2], [2, 2, 1]),
        6: ([3, 3], [2, 2, 2], [2, 3, 1]),
    }
    return list(random.choice(options[n]))


def _layout_objects(
    object_specs: list[dict], legend_shape: tuple[int, int]
) -> tuple[list[dict], list[tuple[int, int]], tuple[int, int], int]:
    specs = [dict(spec) for spec in object_specs]
    n = len(specs)
    counts = _partition_counts(n)
    random.shuffle(specs)

    rows = []
    start = 0
    for count in counts:
        row_specs = specs[start : start + count]
        start += count
        gap_values = [random.randint(1, 2) for _ in range(max(0, len(row_specs) - 1))]
        row_h = max(spec["patch"].shape[0] for spec in row_specs)
        row_w = sum(spec["patch"].shape[1] for spec in row_specs) + sum(gap_values)
        rows.append((row_specs, gap_values, row_h, row_w))

    row_gap_values = [random.randint(1, 2) for _ in range(max(0, len(rows) - 1))]
    block_h = sum(row[2] for row in rows) + sum(row_gap_values)
    block_w = max(row[3] for row in rows)

    legend_top = random.randint(1, 2)
    gap_after_legend = random.randint(2, 4)
    min_h = legend_top + legend_shape[0] + gap_after_legend + block_h + 1
    min_w = max(legend_shape[1] + 2, block_w + 2)
    if min_h > MAX_GRID_SIZE or min_w > MAX_GRID_SIZE:
        raise RuntimeError("layout does not fit")

    grid_h = random.randint(min_h, MAX_GRID_SIZE)
    grid_w = random.randint(min_w, MAX_GRID_SIZE)
    legend_bottom = legend_top + legend_shape[0]
    min_block_top = legend_bottom + gap_after_legend
    max_block_top = grid_h - 1 - block_h
    if min_block_top > max_block_top:
        raise RuntimeError("insufficient vertical slack")
    block_top = random.randint(min_block_top, max_block_top)

    ordered_specs = []
    positions = []
    row_top = block_top
    for row_index, (row_specs, gap_values, row_h, row_w) in enumerate(rows):
        row_left = random.randint(1, grid_w - 1 - row_w)
        col = row_left
        for patch_index, spec in enumerate(row_specs):
            patch = spec["patch"]
            v_off = 0 if patch.shape[0] == row_h else random.randint(0, row_h - patch.shape[0])
            ordered_specs.append(spec)
            positions.append((row_top + v_off, col))
            col += patch.shape[1]
            if patch_index < len(gap_values):
                col += gap_values[patch_index]
        row_top += row_h
        if row_index < len(row_gap_values):
            row_top += row_gap_values[row_index]

    return ordered_specs, positions, (grid_h, grid_w), legend_top


def _make_case() -> tuple[np.ndarray, np.ndarray]:
    entries = _build_lookup_entries()
    lookup = _build_lookup_patch(entries)

    object_specs = []
    base_ids = list(range(len(entries)))
    extras = [random.randrange(len(entries)) for _ in range(random.randint(0, 2))]
    key_ids = base_ids + extras
    random.shuffle(key_ids)
    for key_id in key_ids:
        scale = random.choice((2, 3))
        patch = _scale_patch(entries[key_id]["patch"], scale)
        object_specs.append({"key_id": key_id, "patch": patch})

    ordered_specs, positions, (grid_h, grid_w), legend_top = _layout_objects(object_specs, lookup.shape)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    expected = np.full((grid_h, grid_w), BG, dtype=int)

    legend_left = random.randint(1, grid_w - lookup.shape[1] - 1)
    grid[legend_top : legend_top + lookup.shape[0], legend_left : legend_left + lookup.shape[1]] = lookup
    expected[legend_top : legend_top + lookup.shape[0], legend_left : legend_left + lookup.shape[1]] = lookup

    for spec, (top, left) in zip(ordered_specs, positions, strict=False):
        patch = spec["patch"]
        h, w = patch.shape
        if np.any(grid[top : top + h, left : left + w] != BG):
            raise RuntimeError("object overlap")
        grid[top : top + h, left : left + w] = patch
        expected[top : top + h, left : left + w] = _apply_op(patch, entries[spec["key_id"]]["op"])

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid, expected = _make_case()
        except RuntimeError:
            continue
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
