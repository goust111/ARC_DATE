import random
from collections import Counter

import numpy as np


BG = 0
RED = 2
GRAY = 5
PALETTE = (1, 3, 4, 6, 7, 8, 9)
POINTER_ORDER = ("TL", "TR", "BL", "BR")
GEN_ATTEMPTS = 400
MOTIF_ATTEMPTS = 200
MAX_GRID = 30


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _corner_coord(bbox: tuple[int, int, int, int], corner: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if corner == "TL":
        return r0, c0
    if corner == "TR":
        return r0, c1
    if corner == "BL":
        return r1, c0
    if corner == "BR":
        return r1, c1
    raise ValueError(f"bad corner {corner}")


def _corner_step(corner: str) -> tuple[int, int]:
    if corner == "TL":
        return -1, -1
    if corner == "TR":
        return -1, 1
    if corner == "BL":
        return 1, -1
    if corner == "BR":
        return 1, 1
    raise ValueError(f"bad corner {corner}")


def _in_bbox(bbox: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _outer_shape(orientation: str, inner_h: int, inner_w: int) -> tuple[int, int]:
    if orientation == "H":
        return inner_h + 2, inner_w + 2
    if orientation == "V":
        return inner_w + 2, inner_h + 2
    raise ValueError(f"bad orientation {orientation}")


def _inner_shape(orientation: str, inner_h: int, inner_w: int) -> tuple[int, int]:
    if orientation == "H":
        return inner_h, inner_w
    if orientation == "V":
        return inner_w, inner_h
    raise ValueError(f"bad orientation {orientation}")


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    border_mask = (g == GRAY) | (g == RED)
    comps = _components(border_mask)
    assert comps, "expected at least one frame"

    frames = []
    motif_colors = []
    for comp in comps:
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a non-empty interior"

        expected = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        assert set(comp) == expected, "frame border must be a 1-cell rectangle"

        red_corners = [corner for corner in POINTER_ORDER if g[_corner_coord(bbox, corner)] == RED]
        assert len(red_corners) == 1, "each frame must have exactly one red pointer corner"
        pointer = red_corners[0]
        for corner in POINTER_ORDER:
            rr, cc = _corner_coord(bbox, corner)
            if corner != pointer:
                assert g[rr, cc] == GRAY

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        assert inner.size > 0
        inner_colors = sorted(set(map(int, inner.ravel())) - {BG})
        assert len(inner_colors) == 1, "each frame must contain exactly one motif color"
        motif_color = inner_colors[0]
        motif_mask = inner == motif_color
        local_bbox = _bbox_of_mask(motif_mask)
        assert local_bbox == (0, 0, inner.shape[0] - 1, inner.shape[1] - 1), "motif must fill the full interior bbox"

        inner_h, inner_w = inner.shape
        assert inner_h != inner_w, "frame orientation must be unambiguous"
        orientation = "H" if inner_w > inner_h else "V"

        frames.append(
            {
                "bbox": bbox,
                "pointer": pointer,
                "orientation": orientation,
                "motif_color": motif_color,
                "motif_mask": motif_mask,
            }
        )
        motif_colors.append(motif_color)

    assert len(set(motif_colors)) == len(frames), "motif colors must be unique across frames"
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _find_target_index(frames: list[dict], idx: int, shape: tuple[int, int]) -> int:
    bbox = frames[idx]["bbox"]
    corner = frames[idx]["pointer"]
    r, c = _corner_coord(bbox, corner)
    dr, dc = _corner_step(corner)
    h, w = shape

    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w:
        hit = [j for j, frame in enumerate(frames) if j != idx and _in_bbox(frame["bbox"], rr, cc)]
        if hit:
            assert len(hit) == 1, "ray should enter at most one frame at a time"
            return hit[0]
        rr += dr
        cc += dc
    raise AssertionError("pointer ray must hit another frame")


def _cycle_lengths(targets: list[int]) -> list[int]:
    seen = [False] * len(targets)
    lengths = []
    for start in range(len(targets)):
        if seen[start]:
            continue
        cur = start
        length = 0
        while not seen[cur]:
            seen[cur] = True
            cur = targets[cur]
            length += 1
        lengths.append(length)
    return sorted(lengths)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each frame's interior motif to the nearest frame hit by its red corner pointer, rotating on aspect mismatch.

    Input structure:
    - A black background contains several non-touching gray hollow rectangular frames.
    - Each frame has exactly one red corner pixel on its border; that corner encodes one diagonal ray direction.
    - Inside every frame is a single-color motif mask whose bounding box exactly matches the frame interior, and wide/tall frames use transposed interior sizes.

    Rule:
    - For each frame, follow the diagonal ray starting at its red corner; the first frame bbox entered by that ray is the destination.
    - Clear all frame interiors simultaneously, turn every red pointer corner back to gray, then place each sent motif into its destination interior.
    - If source and destination frames have the same horizontal/vertical orientation, copy the motif as-is; otherwise rotate it 90 degrees clockwise before placing it.

    Key insight:
    - The red corner is an address label rather than part of the motif: it selects a destination frame along a diagonal, and the frame aspect alone decides whether the transferred motif must rotate.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _extract_frames(g)
    targets = [_find_target_index(frames, idx, g.shape) for idx in range(len(frames))]
    assert sorted(targets) == list(range(len(frames))), "send/receive relation must be a permutation"

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = BG
        pr, pc = _corner_coord(frame["bbox"], frame["pointer"])
        out[pr, pc] = GRAY

    for src_idx, dst_idx in enumerate(targets):
        src = frames[src_idx]
        dst = frames[dst_idx]
        patch = src["motif_mask"]
        if src["orientation"] != dst["orientation"]:
            patch = np.rot90(patch, -1)
        expected_shape = (dst["bbox"][2] - dst["bbox"][0] - 1, dst["bbox"][3] - dst["bbox"][1] - 1)
        assert patch.shape == expected_shape, "rotated motif must fit destination interior"

        r0, c0, r1, c1 = dst["bbox"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[patch] = src["motif_color"]

    return out


def _random_motif(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    min_cells = max(h + w - 1, (h * w) // 3)
    max_cells = max(min_cells, h * w - 2)
    for _ in range(MOTIF_ATTEMPTS):
        density = random.uniform(0.38, 0.68)
        mask = np.random.rand(h, w) < density
        cells = int(mask.sum())
        if not (min_cells <= cells <= max_cells):
            continue
        if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    raise RuntimeError("failed to generate a non-degenerate motif")


def _valid_orientation_patterns() -> list[tuple[str, str, str, str]]:
    patterns = []
    for bits in range(16):
        pattern = tuple("V" if (bits >> i) & 1 else "H" for i in range(4))
        if len(set(pattern)) == 1:
            continue
        same = any(pattern[i] == pattern[(i + 1) % 4] for i in range(4))
        diff = any(pattern[i] != pattern[(i + 1) % 4] for i in range(4))
        if same and diff:
            patterns.append(pattern)
    return patterns


ORIENTATION_PATTERNS = _valid_orientation_patterns()


def _build_cycle_layout() -> list[dict] | None:
    inner_h = random.randint(3, 5)
    inner_w = random.randint(inner_h + 2, min(7, inner_h + 4))
    orientations = random.choice(ORIENTATION_PATTERNS)
    outer_shapes = [_outer_shape(ori, inner_h, inner_w) for ori in orientations]

    solutions = []
    for gap1 in range(2, 6):
        for gap2 in range(2, 6):
            ha, wa = outer_shapes[0]
            hb, wb = outer_shapes[1]
            hc, wc = outer_shapes[2]
            hd, wd = outer_shapes[3]
            raw_g3 = hb + wa + 2 * gap1 - hd - wc
            raw_g4 = hb - wa + 2 * gap2 - hd + wc
            if raw_g3 <= 0 or raw_g4 <= 0 or raw_g3 % 2 or raw_g4 % 2:
                continue
            gap3 = raw_g3 // 2
            gap4 = raw_g4 // 2
            if not (1 <= gap3 <= 6 and 1 <= gap4 <= 6):
                continue
            solutions.append((gap1, gap2, gap3, gap4))
    if not solutions:
        return None

    gap1, gap2, gap3, gap4 = random.choice(solutions)
    ha, wa = outer_shapes[0]
    hb, wb = outer_shapes[1]
    hc, wc = outer_shapes[2]
    hd, wd = outer_shapes[3]

    boxes = []
    r0a, c0a = 0, 0
    r1a, c1a = r0a + ha - 1, c0a + wa - 1
    boxes.append((r0a, c0a, r1a, c1a))

    r0b, c0b = r1a + gap1, c1a + gap1
    r1b, c1b = r0b + hb - 1, c0b + wb - 1
    boxes.append((r0b, c0b, r1b, c1b))

    r0c = r1b + gap2
    c1c = c0b - gap2
    r1c = r0c + hc - 1
    c0c = c1c - wc + 1
    boxes.append((r0c, c0c, r1c, c1c))

    r1d = r0c - gap3
    c1d = c0c - gap3
    r0d = r1d - hd + 1
    c0d = c1d - wd + 1
    boxes.append((r0d, c0d, r1d, c1d))

    assert (r0d - gap4, c1d + gap4) == (r1a, c0a)

    min_r = min(r0 for r0, _, _, _ in boxes)
    min_c = min(c0 for _, c0, _, _ in boxes)
    max_r = max(r1 for _, _, r1, _ in boxes)
    max_c = max(c1 for _, _, _, c1 in boxes)
    core_h = max_r - min_r + 1
    core_w = max_c - min_c + 1
    if core_h > MAX_GRID - 2 or core_w > MAX_GRID - 2:
        return None

    top = random.randint(1, min(3, MAX_GRID - core_h))
    left = random.randint(1, min(3, MAX_GRID - core_w))
    bottom_room = MAX_GRID - (core_h + top)
    right_room = MAX_GRID - (core_w + left)
    if bottom_room < 1 or right_room < 1:
        return None
    bottom = random.randint(1, min(3, bottom_room))
    right = random.randint(1, min(3, right_room))
    row_shift = top - min_r
    col_shift = left - min_c

    shifted = []
    for r0, c0, r1, c1 in boxes:
        shifted.append((r0 + row_shift, c0 + col_shift, r1 + row_shift, c1 + col_shift))

    frame_specs = []
    motif_colors = random.sample(PALETTE, 4)
    pointers = ("BR", "BL", "TL", "TR")
    for idx, bbox in enumerate(shifted):
        motif_shape = _inner_shape(orientations[idx], inner_h, inner_w)
        frame_specs.append(
            {
                "bbox": bbox,
                "pointer": pointers[idx],
                "orientation": orientations[idx],
                "motif_color": motif_colors[idx],
                "motif_mask": _random_motif(motif_shape),
            }
        )

    canvas_h = max(r1 for _, _, r1, _ in shifted) + bottom + 1
    canvas_w = max(c1 for _, _, _, c1 in shifted) + right + 1
    if canvas_h > MAX_GRID or canvas_w > MAX_GRID:
        return None
    return frame_specs


def _draw_frames(frame_specs: list[dict]) -> np.ndarray:
    h = max(frame["bbox"][2] for frame in frame_specs) + 1
    w = max(frame["bbox"][3] for frame in frame_specs) + 1
    grid = np.full((h, w), BG, dtype=int)

    for frame in frame_specs:
        r0, c0, r1, c1 = frame["bbox"]
        grid[r0, c0 : c1 + 1] = GRAY
        grid[r1, c0 : c1 + 1] = GRAY
        grid[r0 : r1 + 1, c0] = GRAY
        grid[r0 : r1 + 1, c1] = GRAY
        pr, pc = _corner_coord(frame["bbox"], frame["pointer"])
        grid[pr, pc] = RED

        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        inner[frame["motif_mask"]] = frame["motif_color"]
    return grid


def _apply_random_symmetry(grid: np.ndarray) -> np.ndarray:
    choice = random.randint(0, 7)
    if choice == 0:
        return grid
    if choice == 1:
        return np.rot90(grid, 1)
    if choice == 2:
        return np.rot90(grid, 2)
    if choice == 3:
        return np.rot90(grid, 3)
    if choice == 4:
        return np.fliplr(grid)
    if choice == 5:
        return np.flipud(grid)
    if choice == 6:
        return grid.T
    return np.fliplr(np.rot90(grid, 1))


def _validate_generated(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > MAX_GRID:
        return False
    try:
        frames = _extract_frames(grid)
        if len(frames) != 4:
            return False
        targets = [_find_target_index(frames, idx, grid.shape) for idx in range(len(frames))]
        if sorted(targets) != list(range(len(frames))):
            return False
        if _cycle_lengths(targets) != [4]:
            return False
        if not any(frames[i]["orientation"] == frames[targets[i]]["orientation"] for i in range(4)):
            return False
        if not any(frames[i]["orientation"] != frames[targets[i]]["orientation"] for i in range(4)):
            return False
        out = main(grid)
    except AssertionError:
        return False
    return not np.array_equal(out, grid)


def _gen_attempt() -> np.ndarray | None:
    frame_specs = _build_cycle_layout()
    if frame_specs is None:
        return None
    grid = _draw_frames(frame_specs)
    grid = _apply_random_symmetry(grid).copy()
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 角针快递")
