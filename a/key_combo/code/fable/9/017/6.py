from __future__ import annotations

from collections import Counter

import numpy as np


PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 800
VECTOR_SAMPLE_ATTEMPTS = 400

SHAPE_TEMPLATES = (
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=int),
)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != transparent
    view[mask] = patch[mask]


def collect_objects(panel: np.ndarray, bg: int) -> list[dict]:
    objects: list[dict] = []
    for coords in components_mask(panel != bg):
        bbox = bbox_coords(coords)
        patch = crop_bbox(panel, bbox)
        mask = patch != bg
        objects.append(
            {
                "coords": coords,
                "bbox": bbox,
                "patch": patch,
                "mask": mask,
                "signature": (tuple(int(v) for v in patch.shape), patch.tobytes()),
                "shape_signature": (tuple(int(v) for v in patch.shape), mask.astype(np.uint8).tobytes()),
                "colors": tuple(sorted(map(int, np.unique(patch[mask])))),
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate every sprite from the second panel into a third frame, reversing only the out-of-bounds motion components.

    Input structure:
    - A solid separator column splits the grid into equal-width left and right panels with the same background color.
    - Each panel contains the same set of non-overlapping sprites; each sprite keeps its colored shape and only changes position.
    - Matching sprites are identified by their exact colored patch, not by panel-relative position.

    Rule:
    - Pair every left-panel sprite with the identical colored sprite in the right panel and measure its translation vector.
    - Start from the right-panel position and apply the same vector once more.
    - If that extra step would cross a panel boundary in row and/or column, negate only the offending component(s) before moving, then place the unchanged sprite patch in the output panel.

    Key insight:
    - The two panels are consecutive time frames: solve each object independently, infer its own displacement from frame 1 to frame 2, then continue that motion with per-axis wall bounces.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 2 and width >= 3 and width % 2 == 1, "expected left|bar|right layout"

    mid = width // 2
    separator = g[:, mid]
    assert np.all(separator == separator[0]), "expected a solid separator column"

    left = g[:, :mid]
    right = g[:, mid + 1 :]
    assert left.shape == right.shape and left.size > 0
    bg = get_bg(np.concatenate((left, right), axis=1))
    assert int(separator[0]) != bg, "separator must differ from the panel background"

    left_objects = collect_objects(left, bg)
    right_objects = collect_objects(right, bg)
    assert left_objects and len(left_objects) == len(right_objects), "expected one match per sprite"

    left_by_sig: dict[tuple[tuple[int, ...], bytes], dict] = {}
    right_by_sig: dict[tuple[tuple[int, ...], bytes], dict] = {}
    for obj in left_objects:
        sig = obj["signature"]
        assert sig not in left_by_sig, "left panel contains ambiguous duplicate sprite signatures"
        left_by_sig[sig] = obj
    for obj in right_objects:
        sig = obj["signature"]
        assert sig not in right_by_sig, "right panel contains ambiguous duplicate sprite signatures"
        right_by_sig[sig] = obj
    assert set(left_by_sig) == set(right_by_sig), "panels must contain the same sprite identities"

    panel_h, panel_w = left.shape
    out = np.full((panel_h, panel_w), bg, dtype=int)
    for sig, left_obj in left_by_sig.items():
        right_obj = right_by_sig[sig]
        l_r0, l_c0, l_r1, l_c1 = left_obj["bbox"]
        r_r0, r_c0, r_r1, r_c1 = right_obj["bbox"]
        assert (l_r1 - l_r0, l_c1 - l_c0) == (r_r1 - r_r0, r_c1 - r_c0)

        dr = r_r0 - l_r0
        dc = r_c0 - l_c0
        patch = right_obj["patch"]
        mask = right_obj["mask"]
        h, w = patch.shape

        step_r = dr
        step_c = dc
        if not (0 <= r_r0 + dr and r_r0 + dr + h <= panel_h):
            step_r = -dr
        if not (0 <= r_c0 + dc and r_c0 + dc + w <= panel_w):
            step_c = -dc

        dst_r0 = r_r0 + step_r
        dst_c0 = r_c0 + step_c
        assert 0 <= dst_r0 <= panel_h - h and 0 <= dst_c0 <= panel_w - w

        target = out[dst_r0 : dst_r0 + h, dst_c0 : dst_c0 + w]
        assert not np.any((target != bg) & mask), "extrapolated sprites must not overlap"
        paste(out, patch, dst_r0, dst_c0, transparent=bg)

    return out


def _axis_positions(limit: int, delta: int, bounce: bool) -> list[int]:
    options: list[int] = []
    for start in range(limit + 1):
        first = start + delta
        if not (0 <= first <= limit):
            continue
        second = first + delta
        if bounce:
            if 0 <= second <= limit:
                continue
        else:
            if not (0 <= second <= limit):
                continue
        options.append(start)
    return options


def _sample_vector(
    rng: np.random.Generator,
    panel_h: int,
    panel_w: int,
    patch_shape: tuple[int, int],
    branch: str,
    used: set[tuple[int, int]],
) -> tuple[int, int, list[int], list[int], bool, bool] | None:
    h, w = patch_shape
    row_bounce = branch in {"row", "both"}
    col_bounce = branch in {"col", "both"}
    for _ in range(VECTOR_SAMPLE_ATTEMPTS):
        if branch == "zero":
            dr, dc = 0, 0
        elif branch == "none":
            dr = int(rng.integers(-2, 3))
            dc = int(rng.integers(-2, 3))
            if dr == 0 and dc == 0:
                continue
        elif branch == "row":
            dr = int(rng.choice(np.array([-3, -2, -1, 1, 2, 3], dtype=int)))
            dc = int(rng.integers(-1, 2))
        elif branch == "col":
            dr = int(rng.integers(-1, 2))
            dc = int(rng.choice(np.array([-3, -2, -1, 1, 2, 3], dtype=int)))
        else:
            dr = int(rng.choice(np.array([-2, -1, 1, 2], dtype=int)))
            dc = int(rng.choice(np.array([-2, -1, 1, 2], dtype=int)))

        if (dr, dc) in used:
            continue
        row_positions = _axis_positions(panel_h - h, dr, row_bounce)
        col_positions = _axis_positions(panel_w - w, dc, col_bounce)
        if row_positions and col_positions:
            return dr, dc, row_positions, col_positions, row_bounce, col_bounce
    return None


def _gap_slices(r0: int, c0: int, h: int, w: int, panel_shape: tuple[int, int]) -> tuple[slice, slice]:
    panel_h, panel_w = panel_shape
    rr = slice(max(0, r0 - 1), min(panel_h, r0 + h + 1))
    cc = slice(max(0, c0 - 1), min(panel_w, c0 + w + 1))
    return rr, cc


def _can_place(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr, cc = _gap_slices(r0, c0, h, w, occ.shape)
    return not np.any(occ[rr, cc])


def _mark_place(occ: np.ndarray, r0: int, c0: int, h: int, w: int, delta: int) -> None:
    rr, cc = _gap_slices(r0, c0, h, w, occ.shape)
    occ[rr, cc] += delta


def _colorize(shape: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(shape.shape, bg, dtype=int)
    patch[shape.astype(bool)] = int(color)
    return patch


def _output_top(left_r0: int, left_c0: int, dr: int, dc: int, row_bounce: bool, col_bounce: bool) -> tuple[int, int]:
    out_r0 = left_r0 if row_bounce else left_r0 + 2 * dr
    out_c0 = left_c0 if col_bounce else left_c0 + 2 * dc
    return int(out_r0), int(out_c0)


def _place_sprites(rng: np.random.Generator, panel_shape: tuple[int, int], sprites: list[dict]) -> bool:
    panel_h, panel_w = panel_shape
    left_occ = np.zeros(panel_shape, dtype=int)
    right_occ = np.zeros(panel_shape, dtype=int)
    out_occ = np.zeros(panel_shape, dtype=int)

    for sprite in sprites:
        h, w = sprite["patch"].shape
        candidates = []
        for r0 in sprite["row_positions"]:
            for c0 in sprite["col_positions"]:
                right_r0 = r0 + sprite["dr"]
                right_c0 = c0 + sprite["dc"]
                out_r0, out_c0 = _output_top(
                    r0, c0, sprite["dr"], sprite["dc"], sprite["row_bounce"], sprite["col_bounce"]
                )
                if not (0 <= right_r0 <= panel_h - h and 0 <= right_c0 <= panel_w - w):
                    continue
                if not (0 <= out_r0 <= panel_h - h and 0 <= out_c0 <= panel_w - w):
                    continue
                candidates.append((int(r0), int(c0), int(right_r0), int(right_c0), int(out_r0), int(out_c0)))
        if not candidates:
            return False
        sprite["candidates"] = candidates

    order = sorted(
        range(len(sprites)),
        key=lambda idx: (
            len(sprites[idx]["candidates"]),
            -(sprites[idx]["patch"].shape[0] * sprites[idx]["patch"].shape[1]),
        ),
    )

    def search(depth: int) -> bool:
        if depth == len(order):
            return True
        idx = order[depth]
        sprite = sprites[idx]
        h, w = sprite["patch"].shape
        for pos_idx in rng.permutation(len(sprite["candidates"])):
            left_r0, left_c0, right_r0, right_c0, out_r0, out_c0 = sprite["candidates"][int(pos_idx)]
            if not _can_place(left_occ, left_r0, left_c0, h, w):
                continue
            if not _can_place(right_occ, right_r0, right_c0, h, w):
                continue
            if not _can_place(out_occ, out_r0, out_c0, h, w):
                continue

            _mark_place(left_occ, left_r0, left_c0, h, w, 1)
            _mark_place(right_occ, right_r0, right_c0, h, w, 1)
            _mark_place(out_occ, out_r0, out_c0, h, w, 1)

            sprite["left_top"] = (left_r0, left_c0)
            sprite["right_top"] = (right_r0, right_c0)
            sprite["out_top"] = (out_r0, out_c0)
            if search(depth + 1):
                return True

            _mark_place(left_occ, left_r0, left_c0, h, w, -1)
            _mark_place(right_occ, right_r0, right_c0, h, w, -1)
            _mark_place(out_occ, out_r0, out_c0, h, w, -1)

        return False

    return search(0)


def _render_panel(panel_shape: tuple[int, int], sprites: list[dict], key: str, bg: int) -> np.ndarray:
    panel = np.full(panel_shape, bg, dtype=int)
    for sprite in sprites:
        r0, c0 = sprite[key]
        paste(panel, sprite["patch"], r0, c0, transparent=bg)
    return panel


def _panel_signature(panel: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(int(v) for v in panel.shape), panel.tobytes()


def _validate_case(grid: np.ndarray, expected: np.ndarray, sprites: list[dict], bg: int, separator: int) -> bool:
    height, width = grid.shape
    if height > 30 or width > 30 or max(expected.shape) > 30:
        return False

    mid = width // 2
    if not np.all(grid[:, mid] == separator):
        return False
    if np.any(grid[:, :mid] == separator) or np.any(grid[:, mid + 1 :] == separator):
        return False

    left = grid[:, :mid]
    right = grid[:, mid + 1 :]
    if _panel_signature(left) == _panel_signature(right):
        return False
    if _panel_signature(left) == _panel_signature(expected):
        return False
    if _panel_signature(right) == _panel_signature(expected):
        return False

    vectors = {(int(sprite["dr"]), int(sprite["dc"])) for sprite in sprites}
    if len(vectors) != len(sprites):
        return False
    if not any(sprite["branch"] in {"row", "col"} for sprite in sprites):
        return False
    if not any(sprite["branch"] in {"row", "col", "both"} for sprite in sprites):
        return False
    if not any(sprite["branch"] == "none" for sprite in sprites):
        return False

    right_to_out = {
        (
            int(sprite["out_top"][0] - sprite["right_top"][0]),
            int(sprite["out_top"][1] - sprite["right_top"][1]),
        )
        for sprite in sprites
    }
    if len(right_to_out) < 3:
        return False

    left_objects = collect_objects(left, bg)
    right_objects = collect_objects(right, bg)
    if len(left_objects) != len(sprites) or len(right_objects) != len(sprites):
        return False
    if len({obj["signature"] for obj in left_objects}) != len(sprites):
        return False
    if len({obj["signature"] for obj in right_objects}) != len(sprites):
        return False

    shape_hist = Counter(obj["shape_signature"] for obj in left_objects)
    color_hist = Counter(tuple(obj["colors"]) for obj in left_objects)
    if max(shape_hist.values()) < 2 or max(color_hist.values()) < 2:
        return False

    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    sprite_count = 5 if rng.random() < 0.65 else 4
    panel_h = int(rng.integers(12 if sprite_count == 5 else 11, 15))
    panel_w = int(rng.integers(12 if sprite_count == 5 else 11, 15))

    colors = rng.choice(np.array(PALETTE, dtype=int), size=2 + (4 if sprite_count == 5 else 3), replace=False)
    bg = int(colors[0])
    separator = int(colors[1])
    sprite_colors = [int(v) for v in colors[2:]]

    shape_indices = rng.choice(len(SHAPE_TEMPLATES), size=1 + (3 if sprite_count == 5 else 2), replace=False)
    repeat_shape = SHAPE_TEMPLATES[int(shape_indices[0])]
    extra_shapes = [SHAPE_TEMPLATES[int(idx)] for idx in shape_indices[1:]]

    shared_color = sprite_colors[0]
    other_colors = sprite_colors[1:]
    sprites = [
        {"patch": _colorize(repeat_shape, shared_color, bg), "color": shared_color},
        {"patch": _colorize(extra_shapes[0], shared_color, bg), "color": shared_color},
        {"patch": _colorize(repeat_shape, other_colors[0], bg), "color": other_colors[0]},
        {"patch": _colorize(extra_shapes[1], other_colors[1], bg), "color": other_colors[1]},
    ]
    if sprite_count == 5:
        sprites.append({"patch": _colorize(extra_shapes[2], other_colors[2], bg), "color": other_colors[2]})

    branch_types = ["none", "row", "col", "both"] + (["zero"] if sprite_count == 5 else [])
    rng.shuffle(branch_types)
    rng.shuffle(sprites)
    used_vectors: set[tuple[int, int]] = set()
    for sprite, branch in zip(sprites, branch_types):
        sprite["branch"] = branch
        sampled = _sample_vector(rng, panel_h, panel_w, sprite["patch"].shape, branch, used_vectors)
        if sampled is None:
            return None
        dr, dc, row_positions, col_positions, row_bounce, col_bounce = sampled
        sprite["dr"] = dr
        sprite["dc"] = dc
        sprite["row_positions"] = row_positions
        sprite["col_positions"] = col_positions
        sprite["row_bounce"] = row_bounce
        sprite["col_bounce"] = col_bounce
        used_vectors.add((dr, dc))

    if not _place_sprites(rng, (panel_h, panel_w), sprites):
        return None

    left = _render_panel((panel_h, panel_w), sprites, "left_top", bg)
    right = _render_panel((panel_h, panel_w), sprites, "right_top", bg)
    expected = _render_panel((panel_h, panel_w), sprites, "out_top", bg)
    grid = np.concatenate([left, np.full((panel_h, 1), separator, dtype=int), right], axis=1)

    if not _validate_case(grid, expected, sprites, bg, separator):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
