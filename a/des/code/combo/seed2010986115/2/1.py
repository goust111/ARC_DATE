from __future__ import annotations

import random

import numpy as np


SIDES = ("top", "bottom", "left", "right")
OPPOSITE = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int = 0) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

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
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            rmin = int(arr[:, 0].min())
            rmax = int(arr[:, 0].max())
            cmin = int(arr[:, 1].min())
            cmax = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (rmin, cmin, rmax, cmax),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _u_template(height: int, width: int, missing: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if missing == "top":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[-1, :] = True
    elif missing == "bottom":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[0, :] = True
    elif missing == "left":
        mask[:, -1] = True
        mask[0, :] = True
        mask[-1, :] = True
    elif missing == "right":
        mask[:, 0] = True
        mask[0, :] = True
        mask[-1, :] = True
    else:
        raise ValueError(missing)
    return mask


def _match_frame(comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return None

    mask = np.zeros((height, width), dtype=bool)
    coords = comp["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    for missing in SIDES:
        templ = _u_template(height, width, missing)
        if np.array_equal(mask, templ):
            return {
                "color": int(comp["color"]),
                "bbox": (r0, c0, r1, c1),
                "missing": missing,
            }
    return None


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _find_components(grid, bg=0):
        frame = _match_frame(comp)
        if frame is not None:
            frames.append(frame)
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _find_blocks(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    blocks: list[dict] = []
    for comp in _find_components(g, bg=0):
        r0, c0, r1, c1 = comp["bbox"]
        if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
            continue
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(patch == int(comp["color"])):
            continue
        blocks.append({"color": int(comp["color"]), "bbox": (r0, c0, r1, c1)})
    return blocks


def _block_touches_side(grid: np.ndarray, frame: dict, block: dict, side: str) -> bool:
    g = np.asarray(grid, dtype=int)
    fr0, fc0, fr1, fc1 = frame["bbox"]
    br0, bc0, br1, bc1 = block["bbox"]
    frame_color = int(frame["color"])

    if side == "top":
        return (
            br1 == fr0 - 1
            and bc0 >= fc0
            and bc1 <= fc1
            and np.all(g[fr0, bc0 : bc1 + 1] == frame_color)
        )
    if side == "bottom":
        return (
            br0 == fr1 + 1
            and bc0 >= fc0
            and bc1 <= fc1
            and np.all(g[fr1, bc0 : bc1 + 1] == frame_color)
        )
    if side == "left":
        return (
            bc1 == fc0 - 1
            and br0 >= fr0
            and br1 <= fr1
            and np.all(g[br0 : br1 + 1, fc0] == frame_color)
        )
    if side == "right":
        return (
            bc0 == fc1 + 1
            and br0 >= fr0
            and br1 <= fr1
            and np.all(g[br0 : br1 + 1, fc1] == frame_color)
        )
    raise ValueError(side)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each U-frame's interior with the color of the 2x2 block on the side opposite its opening.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _find_frames(g)
    blocks = _find_blocks(g)
    assert frames, "expected at least one U-shaped frame"

    for frame in frames:
        wanted_side = OPPOSITE[str(frame["missing"])]
        candidates = [block for block in blocks if _block_touches_side(g, frame, block, wanted_side)]
        assert len(candidates) == 1, "each frame must have exactly one opposite-side color block"
        fill_color = int(candidates[0]["color"])
        fr0, fc0, fr1, fc1 = frame["bbox"]
        out[fr0 + 1 : fr1, fc0 + 1 : fc1] = fill_color

    return out


GRID_MIN = 18
GRID_MAX = 30
FRAME_COUNT_CHOICES = (1, 2, 3)
FRAME_SIZE_MIN = 4
FRAME_SIZE_MAX = 8
GEN_RETRIES = 400


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    return out


def _candidate_block_positions(patch: np.ndarray, frame_bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    frame_top, frame_left, frame_bottom, frame_right = frame_bbox
    positions: list[tuple[int, int]] = []

    if side == "top":
        row = frame_top - 2
        for col in range(0, patch.shape[1] - 1):
            if row < 0 or np.any(patch[row : row + 2, col : col + 2] != 0):
                continue
            if np.all(patch[frame_top, col : col + 2] != 0):
                positions.append((row, col))
    elif side == "bottom":
        row = frame_bottom + 1
        for col in range(0, patch.shape[1] - 1):
            if row + 1 >= patch.shape[0] or np.any(patch[row : row + 2, col : col + 2] != 0):
                continue
            if np.all(patch[frame_bottom, col : col + 2] != 0):
                positions.append((row, col))
    elif side == "left":
        col = frame_left - 2
        for row in range(0, patch.shape[0] - 1):
            if col < 0 or np.any(patch[row : row + 2, col : col + 2] != 0):
                continue
            if np.all(patch[row : row + 2, frame_left] != 0):
                positions.append((row, col))
    elif side == "right":
        col = frame_right + 1
        for row in range(0, patch.shape[0] - 1):
            if col + 1 >= patch.shape[1] or np.any(patch[row : row + 2, col : col + 2] != 0):
                continue
            if np.all(patch[row : row + 2, frame_right] != 0):
                positions.append((row, col))
    else:
        raise ValueError(side)

    return positions


def _crop_with_padding(patch: np.ndarray, pad: int = 1) -> tuple[np.ndarray, tuple[int, int]]:
    nonzero = np.argwhere(patch != 0)
    assert nonzero.size > 0
    r0 = max(0, int(nonzero[:, 0].min()) - pad)
    r1 = min(patch.shape[0] - 1, int(nonzero[:, 0].max()) + pad)
    c0 = max(0, int(nonzero[:, 1].min()) - pad)
    c1 = min(patch.shape[1] - 1, int(nonzero[:, 1].max()) + pad)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0)


def _build_frame_patch(
    rng: random.Random,
    height: int,
    width: int,
    missing: str,
    frame_color: int,
    side_colors: dict[str, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    patch = np.zeros((height + 4, width + 4), dtype=int)
    frame_top = 2
    frame_left = 2
    frame_bottom = frame_top + height - 1
    frame_right = frame_left + width - 1
    frame_bbox = (frame_top, frame_left, frame_bottom, frame_right)

    frame_mask = _u_template(height, width, missing)
    patch[frame_top : frame_bottom + 1, frame_left : frame_right + 1][frame_mask] = frame_color

    block_sides = list(side_colors)
    rng.shuffle(block_sides)

    def search(index: int, current: np.ndarray, block_mask: np.ndarray) -> np.ndarray | None:
        if index == len(block_sides):
            return current

        side = block_sides[index]
        candidates = _candidate_block_positions(current, frame_bbox, side)
        rng.shuffle(candidates)
        dilated = _dilate8(block_mask)
        for row, col in candidates:
            if np.any(dilated[row : row + 2, col : col + 2]):
                continue
            nxt = current.copy()
            nxt[row : row + 2, col : col + 2] = int(side_colors[side])
            nxt_mask = block_mask.copy()
            nxt_mask[row : row + 2, col : col + 2] = True
            solved = search(index + 1, nxt, nxt_mask)
            if solved is not None:
                return solved
        return None

    solved_patch = search(0, patch, np.zeros_like(patch, dtype=bool))
    if solved_patch is None:
        return None

    cropped_inp, (crop_r0, crop_c0) = _crop_with_padding(solved_patch, pad=1)
    out_patch = cropped_inp.copy()
    new_top = frame_top - crop_r0
    new_left = frame_left - crop_c0
    out_patch[new_top + 1 : new_top + height - 1, new_left + 1 : new_left + width - 1] = int(
        side_colors[OPPOSITE[missing]]
    )
    return cropped_inp, out_patch


def _place_patches(
    rng: random.Random,
    grid_shape: tuple[int, int],
    patches: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray] | None:
    height, width = grid_shape
    used = np.zeros((height, width), dtype=bool)
    inp = np.zeros((height, width), dtype=int)
    out = np.zeros((height, width), dtype=int)

    order = sorted(range(len(patches)), key=lambda i: patches[i][0].shape[0] * patches[i][0].shape[1], reverse=True)
    for idx in order:
        in_patch, out_patch = patches[idx]
        ph, pw = in_patch.shape
        candidates: list[tuple[int, int]] = []
        for r in range(height - ph + 1):
            for c in range(width - pw + 1):
                if not used[r : r + ph, c : c + pw].any():
                    candidates.append((r, c))
        if not candidates:
            return None
        r0, c0 = rng.choice(candidates)
        inp[r0 : r0 + ph, c0 : c0 + pw] = in_patch
        out[r0 : r0 + ph, c0 : c0 + pw] = out_patch
        used[r0 : r0 + ph, c0 : c0 + pw] = True

    return inp, out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(GRID_MIN, GRID_MAX)
    grid_w = rng.randint(GRID_MIN, GRID_MAX)
    if rng.random() < 0.6:
        grid_w = max(GRID_MIN, min(GRID_MAX, grid_h + rng.randint(-3, 3)))

    frame_count = rng.choice(FRAME_COUNT_CHOICES)
    frame_color = rng.randint(1, 9)

    frame_specs: list[dict] = []
    for _ in range(frame_count):
        frame_specs.append(
            {
                "height": rng.randint(FRAME_SIZE_MIN, FRAME_SIZE_MAX),
                "width": rng.randint(FRAME_SIZE_MIN, FRAME_SIZE_MAX),
                "missing": rng.choice(SIDES),
            }
        )

    distractor_slots: list[tuple[int, str]] = []
    for i, spec in enumerate(frame_specs):
        target_side = OPPOSITE[str(spec["missing"])]
        for side in SIDES:
            if side != spec["missing"] and side != target_side:
                distractor_slots.append((i, side))

    max_extra = min(8 - frame_count, len(distractor_slots))
    extra_count = rng.randint(0, max_extra)
    chosen_distractors = set(rng.sample(distractor_slots, extra_count))

    total_blocks = frame_count + extra_count
    if total_blocks > 8:
        return None

    colors = [color for color in range(1, 10) if color != frame_color]
    rng.shuffle(colors)
    block_colors = colors[:total_blocks]

    color_index = 0
    patches: list[tuple[np.ndarray, np.ndarray]] = []
    for i, spec in enumerate(frame_specs):
        target_side = OPPOSITE[str(spec["missing"])]
        sides = [target_side]
        for side in SIDES:
            if (i, side) in chosen_distractors:
                sides.append(side)
        side_colors: dict[str, int] = {}
        for side in sides:
            side_colors[side] = block_colors[color_index]
            color_index += 1

        patch_pair = _build_frame_patch(
            rng,
            int(spec["height"]),
            int(spec["width"]),
            str(spec["missing"]),
            frame_color,
            side_colors,
        )
        if patch_pair is None:
            return None
        patches.append(patch_pair)

    placed = _place_patches(rng, (grid_h, grid_w), patches)
    if placed is None:
        return None
    inp, expected = placed

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
