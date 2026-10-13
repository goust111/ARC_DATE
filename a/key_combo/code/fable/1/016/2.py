from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
MAX_ATTEMPTS = 400
SPRITE_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 200
PAD = 1
_GEN_CALLS = 0


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def find_components(grid: np.ndarray, *, bg: int) -> list[dict]:
    h, w = grid.shape
    fg = grid != bg
    seen = np.zeros_like(fg, dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not fg[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and fg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coord_arr = np.array(coords, dtype=int)
            components.append(
                {
                    "coords": coord_arr,
                    "bbox": bbox_coords(coord_arr),
                    "area": int(len(coords)),
                }
            )

    components.sort(key=lambda comp: comp["bbox"])
    return components


def sprite_patch(grid: np.ndarray, component: dict, *, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    coords = component["coords"]
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = grid[coords[:, 0], coords[:, 1]]
    return patch


def paste_patch(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, bg: int) -> None:
    r0 = int(r0)
    c0 = int(c0)
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > dst.shape[0] or c0 + w > dst.shape[1]:
        raise AssertionError("patch placement out of bounds")
    window = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != bg
    if np.any(window[mask] != bg):
        raise AssertionError("patch placement overlaps existing foreground")
    window[mask] = patch[mask]


def pattern_key(patch: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tuple(map(int, patch.shape)), tuple(int(x) for x in patch.ravel())


def reflect_patch(patch: np.ndarray, axis_kind: str) -> np.ndarray:
    if axis_kind == "vertical":
        return patch[:, ::-1].copy()
    if axis_kind == "horizontal":
        return patch[::-1, :].copy()
    raise ValueError(f"unknown axis kind: {axis_kind}")


def reflect_bbox(
    box: tuple[int, int, int, int], axis_kind: str, axis_sum: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, box)
    if axis_kind == "vertical":
        return r0, int(axis_sum - c1), r1, int(axis_sum - c0)
    if axis_kind == "horizontal":
        return int(axis_sum - r1), c0, int(axis_sum - r0), c1
    raise ValueError(f"unknown axis kind: {axis_kind}")


def expanded_slices(
    box: tuple[int, int, int, int], shape: tuple[int, int], pad: int = PAD
) -> tuple[slice, slice]:
    h, w = map(int, shape)
    r0, c0, r1, c1 = map(int, box)
    return (
        slice(max(0, r0 - pad), min(h, r1 + pad + 1)),
        slice(max(0, c0 - pad), min(w, c1 + pad + 1)),
    )


def reserve_free(reserve: np.ndarray, box: tuple[int, int, int, int], pad: int = PAD) -> bool:
    rs, cs = expanded_slices(box, reserve.shape, pad=pad)
    return not bool(reserve[rs, cs].any())


def mark_reserve(reserve: np.ndarray, box: tuple[int, int, int, int], pad: int = PAD) -> None:
    rs, cs = expanded_slices(box, reserve.shape, pad=pad)
    reserve[rs, cs] = True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the only repeated sprite crop as a mirror instruction and move every other sprite to its reflected position.

    Input structure:
    - A dominant background contains several separated multicolor 4-connected sprites.
    - Exactly one sprite crop, compared inside its own tight bounding box with background holes preserved, appears twice.
    - The repeated pair is a pure translation either across a row band or across a column band, so its perpendicular bisector lies between two columns or between two rows.
    - Every other sprite stays entirely on one side of that hidden mirror axis and can be reflected in-bounds.

    Rule:
    - Find the unique sprite pattern that occurs exactly twice and use the perpendicular bisector of those two copies as the mirror axis.
    - Keep those two repeated copies unchanged.
    - Remove every other sprite from its original cells and redraw it at the mirrored location across the axis.
    - Reflecting across a vertical axis flips the sprite left-right; reflecting across a horizontal axis flips it top-bottom; colors stay unchanged.

    Key insight:
    - The repeated sprite is not the output target; it is an instruction that defines the hidden mirror line used to relocate every other object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = get_bg(g)

    components = find_components(g, bg=bg)
    assert len(components) >= 3, "expected a repeated pair plus at least one other sprite"

    patches = [sprite_patch(g, comp, bg=bg) for comp in components]
    keys = [pattern_key(patch) for patch in patches]
    counts = Counter(keys)
    duplicate_keys = [key for key, count in counts.items() if count == 2]
    assert len(duplicate_keys) == 1, "expected exactly one repeated sprite pattern"
    assert all(count in (1, 2) for count in counts.values()), "unexpected extra repeated patterns"

    repeated_key = duplicate_keys[0]
    repeated_ids = [idx for idx, key in enumerate(keys) if key == repeated_key]
    assert len(repeated_ids) == 2

    first = components[repeated_ids[0]]
    second = components[repeated_ids[1]]
    patch_h, patch_w = patches[repeated_ids[0]].shape

    same_rows = first["bbox"][0] == second["bbox"][0] and first["bbox"][2] == second["bbox"][2]
    same_cols = first["bbox"][1] == second["bbox"][1] and first["bbox"][3] == second["bbox"][3]
    assert same_rows != same_cols, "repeated pair must align in exactly one direction"

    if same_rows:
        axis_kind = "vertical"
        axis_sum = int(first["bbox"][1] + second["bbox"][1] + patch_w - 1)
        assert axis_sum % 2 == 1, "vertical mirror axis must lie between columns"
    else:
        axis_kind = "horizontal"
        axis_sum = int(first["bbox"][0] + second["bbox"][0] + patch_h - 1)
        assert axis_sum % 2 == 1, "horizontal mirror axis must lie between rows"

    out = np.full_like(g, bg)
    for idx in repeated_ids:
        box = components[idx]["bbox"]
        paste_patch(out, patches[idx], box[0], box[1], bg=bg)

    for idx, comp in enumerate(components):
        if idx in repeated_ids:
            continue
        patch = patches[idx]
        src_box = comp["bbox"]
        dst_box = reflect_bbox(src_box, axis_kind, axis_sum)
        assert dst_box[0] >= 0 and dst_box[1] >= 0
        assert dst_box[2] < g.shape[0] and dst_box[3] < g.shape[1], "mirrored sprite left the grid"
        paste_patch(out, reflect_patch(patch, axis_kind), dst_box[0], dst_box[1], bg=bg)

    return out


def random_connected_patch(
    *,
    min_h: int,
    max_h: int,
    min_w: int,
    max_w: int,
    require_hflip_asym: bool,
) -> np.ndarray:
    for _ in range(SPRITE_ATTEMPTS):
        h = random.randint(min_h, max_h)
        w = random.randint(min_w, max_w)
        target_area = random.randint(max(4, (h * w + 1) // 2), h * w)

        cells = {(random.randrange(h), random.randrange(w))}
        frontier = set()
        frontier.update(
            (nr, nc)
            for nr, nc in neighbors4(*next(iter(cells)))
            if 0 <= nr < h and 0 <= nc < w
        )
        while len(cells) < target_area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nr, nc in neighbors4(*cell):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        if len(cells) != target_area:
            continue

        coords = np.array(sorted(cells), dtype=int)
        r0, c0, r1, c1 = bbox_coords(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

        area = int(mask.sum())
        if area < 4 or mask.shape[0] < 2 or mask.shape[1] < 2:
            continue

        color_count = 2 if area < 6 else random.choice((2, 3))
        colors = random.sample(PALETTE, color_count)
        assignments = [colors[idx % color_count] for idx in range(area)]
        random.shuffle(assignments)

        patch = np.full(mask.shape, BACKGROUND, dtype=int)
        for (rr, cc), color in zip(np.argwhere(mask), assignments):
            patch[int(rr), int(cc)] = int(color)

        if len(set(map(int, patch[patch != BACKGROUND]))) < 2:
            continue
        if require_hflip_asym and np.array_equal(patch, patch[:, ::-1]):
            continue
        return patch

    raise RuntimeError("failed to generate a suitable connected patch")


def build_vertical_scene(source_on_left: bool) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(MAX_ATTEMPTS):
        h = random.randint(14, 20)
        w = random.randint(18, 26)

        pair_patch = random_connected_patch(
            min_h=2,
            max_h=4,
            min_w=2,
            max_w=4,
            require_hflip_asym=False,
        )
        pair_h, pair_w = pair_patch.shape
        pair_gap = random.choice((2, 4))
        axis_col = random.randint(6, w - 7)
        left_c0 = axis_col - pair_gap // 2 - pair_w + 1
        right_c0 = axis_col + 1 + pair_gap // 2
        if left_c0 < 1 or right_c0 + pair_w > w - 1:
            continue

        pair_r0 = random.randint(1, h - pair_h - 2)
        col_sum = left_c0 + right_c0 + pair_w - 1

        grid = np.full((h, w), BACKGROUND, dtype=int)
        expected = np.full((h, w), BACKGROUND, dtype=int)
        input_reserve = np.zeros((h, w), dtype=bool)
        output_reserve = np.zeros((h, w), dtype=bool)

        pair_boxes = [
            (pair_r0, left_c0, pair_r0 + pair_h - 1, left_c0 + pair_w - 1),
            (pair_r0, right_c0, pair_r0 + pair_h - 1, right_c0 + pair_w - 1),
        ]
        for box in pair_boxes:
            paste_patch(grid, pair_patch, box[0], box[1], bg=BACKGROUND)
            paste_patch(expected, pair_patch, box[0], box[1], bg=BACKGROUND)
            mark_reserve(input_reserve, box)
            mark_reserve(output_reserve, box)

        used_keys = {pattern_key(pair_patch)}
        unique_count = random.randint(2, 4)
        placed: list[dict] = []

        for idx in range(unique_count):
            placed_one = False
            for _ in range(PLACEMENT_ATTEMPTS):
                patch = random_connected_patch(
                    min_h=2,
                    max_h=5,
                    min_w=2,
                    max_w=5,
                    require_hflip_asym=(idx == 0),
                )
                key = pattern_key(patch)
                if key in used_keys:
                    continue

                obj_h, obj_w = patch.shape
                if source_on_left:
                    min_c0 = 1
                    max_c0 = axis_col - obj_w
                else:
                    min_c0 = axis_col + 2
                    max_c0 = w - obj_w - 1
                if min_c0 > max_c0:
                    continue

                r0 = random.randint(1, h - obj_h - 2)
                c0 = random.randint(min_c0, max_c0)
                src_box = (r0, c0, r0 + obj_h - 1, c0 + obj_w - 1)
                dst_box = reflect_bbox(src_box, "vertical", col_sum)

                if dst_box[0] < 1 or dst_box[1] < 1 or dst_box[2] > h - 2 or dst_box[3] > w - 2:
                    continue
                if not reserve_free(input_reserve, src_box):
                    continue
                if not reserve_free(output_reserve, dst_box):
                    continue

                paste_patch(grid, patch, src_box[0], src_box[1], bg=BACKGROUND)
                paste_patch(expected, patch[:, ::-1], dst_box[0], dst_box[1], bg=BACKGROUND)
                mark_reserve(input_reserve, src_box)
                mark_reserve(output_reserve, dst_box)
                used_keys.add(key)
                placed.append({"patch": patch, "src_box": src_box, "dst_box": dst_box})
                placed_one = True
                break

            if not placed_one:
                break

        if len(placed) != unique_count:
            continue

        shifts = {entry["dst_box"][1] - entry["src_box"][1] for entry in placed}
        if len(shifts) < 2:
            continue

        try:
            solved = main(grid)
        except AssertionError:
            continue
        if np.array_equal(solved, expected):
            return grid, expected

    raise RuntimeError("failed to build a valid vertical scene")


def generate_input() -> np.ndarray:
    global _GEN_CALLS

    orientation = "vertical" if _GEN_CALLS % 2 == 0 else "horizontal"
    source_on_left = ((_GEN_CALLS // 2) % 2 == 0)
    _GEN_CALLS += 1

    for _ in range(MAX_ATTEMPTS):
        inp, expected = build_vertical_scene(source_on_left=source_on_left)
        if orientation == "horizontal":
            inp = inp.T.copy()
            expected = expected.T.copy()

        out = main(inp)
        if np.array_equal(out, expected):
            return inp

    raise RuntimeError("failed to generate a valid mirror-transport input")
