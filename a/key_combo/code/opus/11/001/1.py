from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

FRAME_H = 7
FRAME_W = 6
INTERIOR_H = FRAME_H - 2
INTERIOR_W = FRAME_W - 2
FRAME_COUNT_CHOICES = (3, 4)
SEPARATOR_CHOICES = (2, 3)
CANVAS_H_RANGE = (18, 24)
FRAME_TOP_RANGE = (1, 4)
HOLE_AREA_RANGE = (4, 8)
MAX_GEN_ATTEMPTS = 400
PALETTE = (BLUE, RED, GREEN, YELLOW)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(arr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        steps = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    canvas[r0 : r0 + h, c0 : c0 + w] = patch


def _render_mask(mask: np.ndarray, color: int) -> np.ndarray:
    sprite = np.zeros(mask.shape, dtype=int)
    sprite[mask] = int(color)
    return sprite


def _frame_record(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    box = (int(r0), int(c0), int(r1), int(c1))
    patch = _crop(grid, box)
    assert patch.shape[0] >= 3 and patch.shape[1] >= 3, "frame too small"
    assert np.all(patch[0, :] == GREY) and np.all(patch[-1, :] == GREY), "frame missing top/bottom border"
    assert np.all(patch[:, 0] == GREY) and np.all(patch[:, -1] == GREY), "frame missing side border"

    hole_mask = patch == BLACK
    hole_components = []
    for comp in _components(hole_mask, connectivity=4):
        if np.all((comp[:, 0] > 0) & (comp[:, 0] < patch.shape[0] - 1) & (comp[:, 1] > 0) & (comp[:, 1] < patch.shape[1] - 1)):
            local_mask = np.zeros_like(hole_mask, dtype=bool)
            local_mask[comp[:, 0], comp[:, 1]] = True
            hole_components.append(local_mask)
    assert len(hole_components) == 1, "expected one interior hole per frame"

    hole_full = hole_components[0]
    hole_box = _bbox(hole_full)
    assert hole_box is not None
    hole_crop = _crop(hole_full, hole_box)
    return {
        "bbox": box,
        "patch": patch,
        "hole_box_rel": hole_box,
        "hole_crop": hole_crop,
    }


def _sprite_record(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    box = (int(r0), int(c0), int(r1), int(c1))
    patch = _crop(grid, box)
    colors = sorted(set(map(int, patch[patch != BLACK])))
    assert len(colors) == 1, "sprite must be monochrome"
    color = colors[0]
    mask = patch != BLACK
    assert patch.shape[1] >= 3, "sprite must have middle section and two peg columns"
    left_rows = np.flatnonzero(mask[:, 0])
    right_rows = np.flatnonzero(mask[:, -1])
    assert len(left_rows) == 1 and len(right_rows) == 1, "sprite must have one left peg and one right peg"
    assert np.count_nonzero(mask[:, 0]) == 1 and np.count_nonzero(mask[:, -1]) == 1
    body_mask = mask[:, 1:-1]
    assert body_mask.any(), "sprite body missing"
    return {
        "bbox": box,
        "color": color,
        "body_mask": body_mask,
        "left_row": int(left_rows[0]),
        "right_row": int(right_rows[0]),
    }


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames = [_frame_record(grid, comp) for comp in _components(grid == GREY, connectivity=4)]
    frames.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    assert 3 <= len(frames) <= 4, "expected 3-4 frames"
    return frames


def _find_sprites(grid: np.ndarray) -> list[dict]:
    mask = (grid != BLACK) & (grid != GREY)
    sprites = [_sprite_record(grid, comp) for comp in _components(mask, connectivity=4)]
    assert sprites, "expected colored sprites"
    return sprites


def _match_frames_and_sprites(frames: list[dict], sprites: list[dict]) -> list[tuple[dict, dict]]:
    matched = []
    used = set()
    for frame in frames:
        candidates = [
            idx
            for idx, sprite in enumerate(sprites)
            if np.array_equal(sprite["body_mask"], frame["hole_crop"])
        ]
        assert len(candidates) == 1, "each hole must match exactly one sprite body"
        idx = candidates[0]
        assert idx not in used, "sprite reused across holes"
        used.add(idx)
        matched.append((frame, sprites[idx]))
    assert len(used) == len(sprites) == len(frames), "matching must be one-to-one"
    return matched


def _embed_frame(frame: dict, sprite: dict) -> tuple[np.ndarray, int, int]:
    patch = frame["patch"].copy()
    hr0, hc0, hr1, hc1 = frame["hole_box_rel"]
    target = patch[hr0 : hr1 + 1, hc0 : hc1 + 1]
    target[sprite["body_mask"]] = int(sprite["color"])

    left_rel = hr0 + int(sprite["left_row"])
    right_rel = hr0 + int(sprite["right_row"])
    patch[left_rel, 0] = int(sprite["color"])
    patch[right_rel, -1] = int(sprite["color"])
    return patch, int(left_rel), int(right_rel)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Embed each colored sprite into its matching hole, then pack the grey frames so peg-defined endpoints align.

    Input structure
    - Black background contains 3-4 grey rectangular frames with solid borders and one black irregular hole inside each frame.
    - Elsewhere, the same number of monochrome colored sprites appear; each sprite equals one hole shape plus exactly one extra pixel on its left edge and one on its right edge.
    - The hole matches the sprite body after removing the sprite's leftmost and rightmost peg columns.

    Rule
    - Match every sprite to the unique frame hole whose tight binary mask equals the sprite body mask.
    - Paste the sprite body into that hole using the sprite color, and place the two peg pixels onto the frame's left and right outer border columns on the corresponding rows.
    - Remove the blank separator columns between frames and keep the leftmost frame at its original vertical position.
    - Then, from left to right, shift each later frame vertically so its left peg row equals the previous placed frame's right peg row, and place it immediately to the right with no gap.

    Key insight
    - The same sprite carries two independent signals: its middle shape says which hole to fill, while its two extra side pixels become the only connector endpoints that determine the packed strip's vertical offsets.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_frames(g)
    sprites = _find_sprites(g)
    matched = _match_frames_and_sprites(frames, sprites)

    embedded = []
    for frame, sprite in matched:
        patch, left_rel, right_rel = _embed_frame(frame, sprite)
        embedded.append(
            {
                "patch": patch,
                "top0": int(frame["bbox"][0]),
                "left_rel": left_rel,
                "right_rel": right_rel,
            }
        )

    tops = [embedded[0]["top0"]]
    for idx in range(1, len(embedded)):
        prev = embedded[idx - 1]
        prev_top = tops[idx - 1]
        prev_right_abs = prev_top + prev["right_rel"]
        tops.append(prev_right_abs - embedded[idx]["left_rel"])

    min_top = min(tops)
    max_bottom = max(top + item["patch"].shape[0] for top, item in zip(tops, embedded))
    out_h = max_bottom - min_top
    out_w = sum(item["patch"].shape[1] for item in embedded)
    out = np.zeros((out_h, out_w), dtype=int)

    c = 0
    for top, item in zip(tops, embedded):
        patch = item["patch"]
        r = top - min_top
        _paste(out, patch, r, c)
        c += patch.shape[1]

    return out


def _build_hole_library() -> list[dict]:
    library = []
    for h in range(2, INTERIOR_H + 1):
        for bits in range(1 << (h * INTERIOR_W)):
            area = bits.bit_count()
            if not (HOLE_AREA_RANGE[0] <= area <= HOLE_AREA_RANGE[1]):
                continue
            mask = np.zeros((h, INTERIOR_W), dtype=bool)
            for bit in range(h * INTERIOR_W):
                if bits & (1 << bit):
                    mask[bit // INTERIOR_W, bit % INTERIOR_W] = True
            if not mask[0, :].any() or not mask[-1, :].any():
                continue
            if not mask[:, 0].any() or not mask[:, -1].any():
                continue
            if len(_components(mask, connectivity=4)) != 1:
                continue
            left_rows = tuple(int(r) for r in np.flatnonzero(mask[:, 0]))
            right_rows = tuple(int(r) for r in np.flatnonzero(mask[:, -1]))
            if len(left_rows) < 2 or len(right_rows) < 2:
                continue
            library.append(
                {
                    "mask": mask,
                    "area": area,
                    "height": h,
                    "left_rows": left_rows,
                    "right_rows": right_rows,
                }
            )
    return library


HOLE_LIBRARY = _build_hole_library()


def _sample_holes(n_frames: int) -> list[dict] | None:
    for _ in range(200):
        items = random.sample(HOLE_LIBRARY, n_frames)
        signature_counts = Counter((item["height"], item["area"]) for item in items)
        if max(signature_counts.values()) < 2:
            continue
        return items
    return None


def _assignment_options(item: dict) -> list[dict]:
    h = item["height"]
    options = []
    for top_offset in range(INTERIOR_H - h + 1):
        for left_local in item["left_rows"]:
            for right_local in item["right_rows"]:
                options.append(
                    {
                        "top_offset": int(top_offset),
                        "left_local": int(left_local),
                        "right_local": int(right_local),
                        "left_rel": int(1 + top_offset + left_local),
                        "right_rel": int(1 + top_offset + right_local),
                    }
                )
    random.shuffle(options)
    return options


def _assign_endpoints(items: list[dict]) -> list[dict] | None:
    options = [_assignment_options(item) for item in items]
    n = len(items)

    def dfs(idx: int, chosen: list[dict], used_left: set[int], used_right: set[int], shifts: list[int]) -> list[dict] | None:
        if idx == n:
            return list(chosen)
        for option in options[idx]:
            if option["left_rel"] in used_left or option["right_rel"] in used_right:
                continue
            if idx > 0:
                shift = chosen[-1]["right_rel"] - option["left_rel"]
                if shift == 0 or shift in shifts:
                    continue
            else:
                shift = None

            chosen.append(option)
            used_left.add(option["left_rel"])
            used_right.add(option["right_rel"])
            if shift is not None:
                shifts.append(shift)

            result = dfs(idx + 1, chosen, used_left, used_right, shifts)
            if result is not None:
                return result

            chosen.pop()
            used_left.remove(option["left_rel"])
            used_right.remove(option["right_rel"])
            if shift is not None:
                shifts.pop()
        return None

    return dfs(0, [], set(), set(), [])


def _frame_patch(mask: np.ndarray, top_offset: int) -> np.ndarray:
    patch = np.full((FRAME_H, FRAME_W), GREY, dtype=int)
    h = mask.shape[0]
    hole = patch[1 + top_offset : 1 + top_offset + h, 1 : 1 + INTERIOR_W]
    hole[mask] = BLACK
    return patch


def _sprite_patch(mask: np.ndarray, color: int, left_local: int, right_local: int) -> np.ndarray:
    h = mask.shape[0]
    patch = np.zeros((h, INTERIOR_W + 2), dtype=int)
    patch[:, 1:-1][mask] = int(color)
    patch[int(left_local), 0] = int(color)
    patch[int(right_local), -1] = int(color)
    return patch


def _embedded_patch(mask: np.ndarray, color: int, top_offset: int, left_local: int, right_local: int) -> tuple[np.ndarray, int, int]:
    patch = _frame_patch(mask, top_offset)
    h = mask.shape[0]
    band = patch[1 + top_offset : 1 + top_offset + h, 1 : 1 + INTERIOR_W]
    band[mask] = int(color)
    left_rel = 1 + top_offset + int(left_local)
    right_rel = 1 + top_offset + int(right_local)
    patch[left_rel, 0] = int(color)
    patch[right_rel, -1] = int(color)
    return patch, left_rel, right_rel


def _expected_from_specs(frame_specs: list[dict], frame_top: int) -> np.ndarray:
    tops = [int(frame_top)]
    for idx in range(1, len(frame_specs)):
        prev = frame_specs[idx - 1]
        prev_top = tops[idx - 1]
        prev_right_abs = prev_top + prev["right_rel"]
        tops.append(prev_right_abs - frame_specs[idx]["left_rel"])

    min_top = min(tops)
    max_bottom = max(top + FRAME_H for top in tops)
    out = np.zeros((max_bottom - min_top, FRAME_W * len(frame_specs)), dtype=int)
    c = 0
    for top, spec in zip(tops, frame_specs):
        _paste(out, spec["embedded_patch"], top - min_top, c)
        c += FRAME_W
    return out


def _place_sprites(input_grid: np.ndarray, sprite_specs: list[dict], frame_cols: list[int], frame_bottom: int) -> bool:
    h, _w = input_grid.shape
    zone_top = frame_bottom + 2
    perm = list(range(len(sprite_specs)))
    random.shuffle(perm)
    if len(perm) > 1 and perm == list(range(len(sprite_specs))):
        perm = perm[1:] + perm[:1]

    for sprite_idx, band_idx in enumerate(perm):
        sprite = sprite_specs[sprite_idx]["sprite_patch"]
        sh, sw = sprite.shape
        col = frame_cols[band_idx]
        row_choices = list(range(zone_top, h - sh + 1))
        if not row_choices:
            return False
        row = random.choice(row_choices)
        target = input_grid[row : row + sh, col : col + sw]
        if np.any(target != BLACK):
            return False
        target[:] = sprite
    return True


def _separator_columns(frame_cols: list[int], n_frames: int, seps: list[int]) -> list[int]:
    cols = []
    c = frame_cols[0] + FRAME_W
    for idx in range(n_frames - 1):
        cols.extend(range(c, c + seps[idx]))
        c += seps[idx] + FRAME_W
    return cols


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, frame_specs: list[dict], seps: list[int]) -> bool:
    if not (inp.ndim == 2 and expected.ndim == 2):
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    if not np.array_equal(main(inp), expected):
        return False

    frame_cols = []
    c = 0
    for idx in range(len(frame_specs)):
        frame_cols.append(c)
        c += FRAME_W
        if idx < len(seps):
            c += seps[idx]

    for col in _separator_columns(frame_cols, len(frame_specs), seps):
        if not np.all(inp[:, col] == BLACK):
            return False

    holes = [spec["mask"] for spec in frame_specs]
    if len({hole.tobytes() for hole in holes}) != len(holes):
        return False

    dims_areas = Counter((spec["mask"].shape[0], spec["mask"].sum()) for spec in frame_specs)
    if max(dims_areas.values()) < 2:
        return False

    for spec in frame_specs:
        left_choices = {1 + spec["top_offset"] + r for r in spec["left_rows"]}
        right_choices = {1 + spec["top_offset"] + r for r in spec["right_rows"]}
        if len(left_choices) < 2 or len(right_choices) < 2:
            return False

    shifts = [frame_specs[i - 1]["right_rel"] - frame_specs[i]["left_rel"] for i in range(1, len(frame_specs))]
    if any(shift == 0 for shift in shifts) or len(set(shifts)) != len(shifts):
        return False

    frames = _find_frames(inp)
    sprites = _find_sprites(inp)
    if len(frames) != len(frame_specs) or len(sprites) != len(frame_specs):
        return False
    for frame in frames:
        count = sum(np.array_equal(frame["hole_crop"], sprite["body_mask"]) for sprite in sprites)
        if count != 1:
            return False
    return True


def _attempt_generate() -> np.ndarray | None:
    n_frames = random.choice(FRAME_COUNT_CHOICES)
    holes = _sample_holes(n_frames)
    if holes is None:
        return None
    assignments = _assign_endpoints(holes)
    if assignments is None:
        return None

    if n_frames == 4:
        seps = [2, 2, 2]
    else:
        seps = [random.choice(SEPARATOR_CHOICES) for _ in range(n_frames - 1)]

    width = n_frames * FRAME_W + sum(seps)
    if width > 30:
        return None

    frame_top = random.randint(*FRAME_TOP_RANGE)
    max_sprite_h = max(item["height"] for item in holes)
    min_h = max(CANVAS_H_RANGE[0], frame_top + FRAME_H + 2 + max_sprite_h)
    if min_h > CANVAS_H_RANGE[1]:
        return None
    canvas_h = random.randint(min_h, CANVAS_H_RANGE[1])

    colors = random.sample(PALETTE, n_frames)
    frame_specs = []
    for hole, assignment, color in zip(holes, assignments, colors):
        sprite = _sprite_patch(hole["mask"], color, assignment["left_local"], assignment["right_local"])
        embedded_patch, left_rel, right_rel = _embedded_patch(
            hole["mask"],
            color,
            assignment["top_offset"],
            assignment["left_local"],
            assignment["right_local"],
        )
        frame_specs.append(
            {
                "mask": hole["mask"],
                "area": hole["area"],
                "height": hole["height"],
                "left_rows": hole["left_rows"],
                "right_rows": hole["right_rows"],
                "top_offset": assignment["top_offset"],
                "left_rel": left_rel,
                "right_rel": right_rel,
                "embedded_patch": embedded_patch,
                "sprite_patch": sprite,
            }
        )

    inp = np.zeros((canvas_h, width), dtype=int)
    frame_cols = []
    col = 0
    for idx, spec in enumerate(frame_specs):
        frame_cols.append(col)
        _paste(inp, _frame_patch(spec["mask"], spec["top_offset"]), frame_top, col)
        col += FRAME_W
        if idx < len(seps):
            col += seps[idx]

    if not _place_sprites(inp, frame_specs, frame_cols, frame_top + FRAME_H - 1):
        return None

    expected = _expected_from_specs(frame_specs, frame_top)
    if not _semantic_checks(inp, expected, frame_specs, seps):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
