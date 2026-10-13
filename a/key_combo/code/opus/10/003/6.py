from __future__ import annotations

from collections import Counter, deque
import itertools
import random

import numpy as np


BG = 0
PLACEHOLDER = 5
LABEL_COLORS = (1, 2, 3)
SPRITE_COLORS = (4, 6, 7, 8, 9)
ROTATION_BY_DISTINCT_COUNT = {
    1: 0,  # 0 degrees
    2: 3,  # clockwise 90 degrees
    3: 2,  # 180 degrees
}

GENERATION_ATTEMPTS = 500
GRID_HEIGHT_RANGE = (14, 22)
GRID_WIDTH_RANGE = (18, 28)
RECT_PADDING = 2


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste_transparent(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int = BG) -> None:
    r0 = int(r0)
    c0 = int(c0)
    h, w = patch.shape
    dst_view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(transparent)
    dst_view[mask] = patch[mask]


def _component_from_seed(mask: np.ndarray, start: tuple[int, int], visited: np.ndarray) -> np.ndarray:
    q: deque[tuple[int, int]] = deque([start])
    visited[start] = True
    coords: list[tuple[int, int]] = []
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        coords.append((int(r), int(c)))
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                q.append((int(nr), int(nc)))
    return np.array(coords, dtype=int)


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    monochromatic: bool = True,
    colors: tuple[int, ...] | list[int] | None = None,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = get_bg(g)
    bg = int(bg)

    if monochromatic:
        if colors is None:
            color_list = sorted(set(map(int, g.flatten())) - {bg})
        else:
            color_list = [int(c) for c in colors if int(c) != bg]
        out: list[dict] = []
        for color in color_list:
            mask = g == int(color)
            visited = np.zeros_like(mask, dtype=bool)
            for r, c in np.argwhere(mask):
                rr = int(r)
                cc = int(c)
                if visited[rr, cc]:
                    continue
                coords = _component_from_seed(mask, (rr, cc), visited)
                out.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": bbox_coords(coords),
                        "area": int(coords.shape[0]),
                    }
                )
        return out

    if colors is None:
        mask = g != bg
    else:
        mask = (g != bg) & np.isin(g, list(map(int, colors)))
    visited = np.zeros_like(mask, dtype=bool)
    out = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if visited[rr, cc]:
            continue
        coords = _component_from_seed(mask, (rr, cc), visited)
        out.append(
            {
                "color": None,
                "coords": coords,
                "bbox": bbox_coords(coords),
                "area": int(coords.shape[0]),
            }
        )
    return out


def _local_monochrome_mask(grid: np.ndarray, comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    color = int(comp["color"])
    return grid[r0 : r1 + 1, c0 : c1 + 1] == color


def _local_foreground_mask(grid: np.ndarray, comp: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return patch != int(bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the distinct-color count in each right-side tag to choose how the source sprite is rotated before stamping it onto each placeholder.

    Input structure:
    - The background is black.
    - One polychromatic source sprite appears as a single 4-connected object.
    - Three grey placeholder objects share the source sprite's normalized mask.
    - Each placeholder has a 3x3 colored tag region immediately to the right of its bbox.

    Rule:
    - Identify the common grey placeholder mask and the unique colored source sprite with the same mask.
    - For each placeholder, count the distinct non-black colors in its adjacent 3x3 tag.
    - Count 1 means stamp the source with no rotation, count 2 means rotate it clockwise by 90 degrees, and count 3 means rotate it by 180 degrees.
    - Erase the original source sprite and every tag region, clear each placeholder bbox, then paste the rotated source patch aligned to that placeholder bbox's top-left corner.

    Key insight:
    - The tags are not decorative; only their number of distinct colors matters, and that count encodes the rotation applied to the source stamp.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    placeholders = find_components(g, bg=bg, monochromatic=True, colors=[PLACEHOLDER])
    assert len(placeholders) == 3, "expected exactly three grey placeholders"

    placeholder_masks = [_local_monochrome_mask(g, comp) for comp in placeholders]
    placeholder_mask = placeholder_masks[0]
    assert all(mask.shape == placeholder_mask.shape and np.array_equal(mask, placeholder_mask) for mask in placeholder_masks)

    source_candidates: list[tuple[np.ndarray, dict]] = []
    for comp in find_components(g, bg=bg, monochromatic=False):
        patch = crop_bbox(g, comp["bbox"])
        mask = _local_foreground_mask(g, comp, bg)
        colors = set(map(int, patch[mask]))
        if PLACEHOLDER in colors:
            continue
        if len(colors) < 2:
            continue
        if patch.shape == placeholder_mask.shape and np.array_equal(mask, placeholder_mask):
            source_candidates.append((patch, comp))

    assert len(source_candidates) == 1, "expected a unique source sprite"
    source_patch, source_comp = source_candidates[0]

    out = g.copy()
    source_coords = source_comp["coords"]
    out[source_coords[:, 0], source_coords[:, 1]] = bg

    for comp in placeholders:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        tag = g[r0 : r0 + 3, c1 + 1 : c1 + 4]
        assert tag.shape == (3, 3), "every placeholder must have a 3x3 tag to its right"
        distinct = {int(v) for v in tag.flatten() if int(v) != bg}
        assert len(distinct) in ROTATION_BY_DISTINCT_COUNT, "tag color count must be 1, 2, or 3"

        out[r0 : r0 + 3, c1 + 1 : c1 + 4] = bg
        out[r0 : r1 + 1, c0 : c1 + 1] = bg
        rotated = np.rot90(source_patch, ROTATION_BY_DISTINCT_COUNT[len(distinct)])
        paste_transparent(out, rotated, r0, c0, transparent=bg)

    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    visited = np.zeros_like(mask, dtype=bool)
    comp = _component_from_seed(mask, tuple(map(int, coords[0])), visited)
    return int(comp.shape[0]) == int(mask.sum())


def _build_mask_options() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    cells = list(range(9))
    for area in range(4, 8):
        for combo in itertools.combinations(cells, area):
            mask = np.zeros((3, 3), dtype=bool)
            for idx in combo:
                mask[idx // 3, idx % 3] = True
            if not _is_connected(mask):
                continue
            if not mask[0].any() or not mask[2].any() or not mask[:, 0].any() or not mask[:, 2].any():
                continue
            if len({np.rot90(mask, k).tobytes() for k in (0, 1, 2)}) < 3:
                continue
            masks.append(mask)
    return masks


MASK_OPTIONS = _build_mask_options()


def _make_sprite(mask: np.ndarray) -> np.ndarray:
    coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
    color_count = random.randint(2, 3)
    palette = random.sample(SPRITE_COLORS, color_count)
    while True:
        patch = np.zeros(mask.shape, dtype=int)
        ordered = coords.copy()
        random.shuffle(ordered)
        for i, (r, c) in enumerate(ordered):
            patch[r, c] = palette[i] if i < color_count else random.choice(palette)
        if len(set(map(int, patch[mask]))) < 2:
            continue
        if len({np.rot90(patch, k).tobytes() for k in (0, 1, 2)}) < 3:
            continue
        return patch


def _make_tag(distinct_count: int) -> np.ndarray:
    assert distinct_count in (1, 2, 3)
    chosen = random.sample(LABEL_COLORS, distinct_count)
    flat = np.empty(9, dtype=int)
    flat[:distinct_count] = np.array(chosen, dtype=int)
    flat[distinct_count:] = np.random.choice(chosen, size=9 - distinct_count, replace=True)
    np.random.shuffle(flat)
    return flat.reshape(3, 3)


def _rect_clear(existing: list[tuple[int, int, int, int]], rect: tuple[int, int, int, int], padding: int) -> bool:
    r0, c0, r1, c1 = map(int, rect)
    for er0, ec0, er1, ec1 in existing:
        if not (r1 + padding < er0 or er1 + padding < r0 or c1 + padding < ec0 or ec1 + padding < c0):
            return False
    return True


def _place_rectangles(shape: tuple[int, int], sizes: list[tuple[int, int]], padding: int) -> list[tuple[int, int]] | None:
    h, w = map(int, shape)
    sizes_with_index = sorted(enumerate(sizes), key=lambda item: item[1][0] * item[1][1], reverse=True)
    placements: list[tuple[int, int] | None] = [None] * len(sizes)
    occupied: list[tuple[int, int, int, int]] = []

    def search(i: int) -> bool:
        if i == len(sizes_with_index):
            return True
        original_idx, (rh, rw) = sizes_with_index[i]
        candidates = [(r, c) for r in range(h - rh + 1) for c in range(w - rw + 1)]
        random.shuffle(candidates)
        for r0, c0 in candidates:
            rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if not _rect_clear(occupied, rect, padding):
                continue
            occupied.append(rect)
            placements[original_idx] = (int(r0), int(c0))
            if search(i + 1):
                return True
            occupied.pop()
            placements[original_idx] = None
        return False

    if not search(0):
        return None
    return [pos for pos in placements if pos is not None]


def _expected_output(
    inp: np.ndarray,
    source_pos: tuple[int, int],
    source_patch: np.ndarray,
    placeholders: list[dict],
) -> np.ndarray:
    out = np.array(inp, dtype=int)
    sr, sc = map(int, source_pos)
    source_mask = source_patch != BG
    out[sr : sr + source_patch.shape[0], sc : sc + source_patch.shape[1]][source_mask] = BG
    for info in placeholders:
        r0 = int(info["r"])
        c0 = int(info["c"])
        out[r0 : r0 + 3, c0 + 3 : c0 + 6] = BG
        out[r0 : r0 + 3, c0 : c0 + 3] = BG
        rotated = np.rot90(source_patch, ROTATION_BY_DISTINCT_COUNT[int(info["count"])])
        paste_transparent(out, rotated, r0, c0, transparent=BG)
    return out


def _semantic_checks(inp: np.ndarray, out: np.ndarray, source_patch: np.ndarray, placeholders: list[dict]) -> bool:
    if sorted(info["count"] for info in placeholders) != [1, 2, 3]:
        return False
    if len({np.rot90(source_patch, k).tobytes() for k in (0, 1, 2)}) != 3:
        return False

    stamped_patches = []
    for info in placeholders:
        r0 = int(info["r"])
        c0 = int(info["c"])
        stamped_patches.append(out[r0 : r0 + 3, c0 : c0 + 3].copy())
    if len({patch.tobytes() for patch in stamped_patches}) != 3:
        return False

    unrotated = source_patch.tobytes()
    if not any(np.rot90(source_patch, ROTATION_BY_DISTINCT_COUNT[info["count"]]).tobytes() != unrotated for info in placeholders):
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(solved, out))


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_HEIGHT_RANGE)
    grid_w = random.randint(*GRID_WIDTH_RANGE)
    placements = _place_rectangles((grid_h, grid_w), [(3, 3), (3, 6), (3, 6), (3, 6)], RECT_PADDING)
    if placements is None:
        return None

    sprite = _make_sprite(random.choice(MASK_OPTIONS))
    sprite = np.rot90(sprite, random.randint(0, 3))

    counts = [1, 2, 3]
    random.shuffle(counts)
    placeholder_infos = []
    for idx, (r0, c0) in enumerate(placements[1:]):
        placeholder_infos.append({"r": int(r0), "c": int(c0), "count": int(counts[idx]), "tag": _make_tag(int(counts[idx]))})

    inp = np.zeros((grid_h, grid_w), dtype=int)
    sr, sc = map(int, placements[0])
    paste_transparent(inp, sprite, sr, sc, transparent=BG)

    placeholder_mask = sprite != BG
    grey_patch = np.zeros_like(sprite)
    grey_patch[placeholder_mask] = PLACEHOLDER
    for info in placeholder_infos:
        r0 = int(info["r"])
        c0 = int(info["c"])
        paste_transparent(inp, grey_patch, r0, c0, transparent=BG)
        inp[r0 : r0 + 3, c0 + 3 : c0 + 6] = info["tag"]

    expected = _expected_output(inp, (sr, sc), sprite, placeholder_infos)
    if not _semantic_checks(inp, expected, sprite, placeholder_infos):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
