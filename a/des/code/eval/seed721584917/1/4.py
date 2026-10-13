from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
ORANGE = 7
PALETTE_POOL = [1, 3, 4, 6, 8, 9]
MAX_GENERATION_ATTEMPTS = 400

PATTERNS = [
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 1]], dtype=bool),
]

CORNER_TO_TURNS = {
    "tl": 0,
    "tr": 1,
    "br": 2,
    "bl": 3,
}


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
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
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _draw_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _corner_name(bbox: tuple[int, int, int, int], cell: tuple[int, int]) -> str:
    r0, c0, r1, c1 = bbox
    mapping = {
        (r0, c0): "tl",
        (r0, c1): "tr",
        (r1, c1): "br",
        (r1, c0): "bl",
    }
    assert cell in mapping, "marker pixel must be at a frame corner"
    return mapping[cell]


def _corner_cell(bbox: tuple[int, int, int, int], turns_cw: int) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    mapping = {
        0: (r0, c0),
        1: (r0, c1),
        2: (r1, c1),
        3: (r1, c0),
    }
    return mapping[int(turns_cw)]


def _rotate_mask(mask: np.ndarray, turns_cw: int) -> np.ndarray:
    return np.rot90(np.asarray(mask, dtype=bool), -int(turns_cw))


def _center_offset(inner_h: int, inner_w: int, shape: tuple[int, int]) -> tuple[int, int]:
    ph, pw = map(int, shape)
    assert ph <= inner_h and pw <= inner_w
    return (inner_h - ph) // 2, (inner_w - pw) // 2


def _parse_palette(bottom_row: np.ndarray) -> list[int]:
    row = np.asarray(bottom_row, dtype=int)
    assert row.ndim == 1 and row.size > 0
    assert not np.any(row == BLACK), "palette row must be fully colored"
    colors = [int(row[0])]
    for value in row[1:]:
        color = int(value)
        if color != colors[-1]:
            colors.append(color)
    assert len(colors) == len(set(colors)), "palette blocks must stay visually distinct"
    return colors


def _detect_frames(grid: np.ndarray) -> list[dict]:
    mask = (grid == ORANGE) | (grid == RED)
    frames: list[dict] = []
    for coords in _find_components(mask):
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 3 and width >= 3

        comp_mask = np.zeros_like(mask, dtype=bool)
        comp_mask[coords[:, 0], coords[:, 1]] = True
        expected = np.zeros((height, width), dtype=bool)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        expected[:, -1] = True
        assert np.array_equal(comp_mask[r0 : r1 + 1, c0 : c1 + 1], expected), "component must be a hollow frame"

        values = grid[coords[:, 0], coords[:, 1]]
        marker_cells = coords[values == RED]
        assert len(marker_cells) == 1, "each frame must have exactly one red corner marker"
        marker = tuple(map(int, marker_cells[0]))
        turns_cw = CORNER_TO_TURNS[_corner_name(bbox, marker)]
        frames.append({"bbox": bbox, "orientation": turns_cw})

    assert frames, "expected at least one frame"
    return frames


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _source_pattern(grid: np.ndarray, frame: dict) -> np.ndarray:
    r0, c0, r1, c1 = frame["bbox"]
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert set(map(int, np.unique(interior))) <= {BLACK, GREY}

    grey_coords = np.argwhere(interior == GREY)
    assert len(grey_coords) > 0, "innermost frame must contain a grey source pattern"
    pr0, pc0 = grey_coords.min(axis=0)
    pr1, pc1 = grey_coords.max(axis=0)
    pattern = interior[pr0 : pr1 + 1, pc0 : pc1 + 1] == GREY
    assert pattern.any()
    return pattern


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = map(int, g.shape)
    assert height >= 4 and width >= 4

    palette = _parse_palette(g[-1])
    board = g[:-1]
    frames = _detect_frames(board)

    for idx, frame in enumerate(frames):
        parents = [
            parent_idx
            for parent_idx, other in enumerate(frames)
            if parent_idx != idx and _contains(other["bbox"], frame["bbox"])
        ]
        frame["depth"] = len(parents)
        frame["parent"] = min(parents, key=lambda parent_idx: _bbox_area(frames[parent_idx]["bbox"])) if parents else None
        frame["children"] = []

    for idx, frame in enumerate(frames):
        parent = frame["parent"]
        if parent is not None:
            frames[parent]["children"].append(idx)

    max_depth = max(int(frame["depth"]) for frame in frames)
    assert len(palette) >= max_depth + 1, "palette row must provide one color per depth level"

    leaf_patterns: dict[int, np.ndarray] = {}
    for idx, frame in enumerate(frames):
        if not frame["children"]:
            leaf_patterns[idx] = _source_pattern(board, frame)

    def unique_leaf(idx: int) -> int:
        stack = [idx]
        leaves: list[int] = []
        while stack:
            current = stack.pop()
            children = frames[current]["children"]
            if not children:
                leaves.append(current)
            else:
                stack.extend(children)
        assert len(leaves) == 1, "each frame must lead to a unique innermost pattern"
        return leaves[0]

    out = np.full_like(board, BLACK)
    for frame in frames:
        color = int(palette[int(frame["depth"])])
        _draw_border(out, frame["bbox"], color)

    for idx, frame in enumerate(frames):
        leaf_idx = unique_leaf(idx)
        color = GREY if idx == leaf_idx else int(palette[int(frame["depth"])])
        rotated = _rotate_mask(leaf_patterns[leaf_idx], int(frame["orientation"]))
        r0, c0, r1, c1 = frame["bbox"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        off_r, off_c = _center_offset(inner_h, inner_w, rotated.shape)
        top = r0 + 1 + off_r
        left = c0 + 1 + off_c
        patch = out[top : top + rotated.shape[0], left : left + rotated.shape[1]]
        assert patch.shape == rotated.shape
        assert not np.any(patch[rotated] != BLACK), "pattern copy must stay inside free interior space"
        patch[rotated] = color

    return out


def _rects_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 0) -> bool:
    ar0, ac0, ah, aw = a
    br0, bc0, bh, bw = b
    ar1 = ar0 + ah - 1
    ac1 = ac0 + aw - 1
    br1 = br0 + bh - 1
    bc1 = bc0 + bw - 1
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _inner_structure(pattern: np.ndarray, orientation: int) -> dict:
    base_h, base_w = map(int, pattern.shape)
    rot_h, rot_w = map(int, _rotate_mask(pattern, orientation).shape)
    inner_h = random.randint(max(base_h, rot_h) + 1, max(base_h, rot_h) + 2)
    inner_w = random.randint(max(base_w, rot_w) + 1, max(base_w, rot_w) + 2)
    return {
        "height": inner_h + 2,
        "width": inner_w + 2,
        "frames": [{"bbox": (0, 0, inner_h + 1, inner_w + 1), "orientation": int(orientation)}],
    }


def _wrap_structure(structure: dict, pattern: np.ndarray, orientation: int) -> dict | None:
    child_h = int(structure["height"])
    child_w = int(structure["width"])
    pat_h, pat_w = map(int, _rotate_mask(pattern, orientation).shape)
    quadrants = ("nw", "ne", "sw", "se")

    for _ in range(200):
        quadrant = random.choice(quadrants)
        small = lambda: random.randint(1, 2)
        large = lambda: random.randint(8, 14)
        if quadrant == "nw":
            top_pad, left_pad, bottom_pad, right_pad = small(), small(), large(), large()
        elif quadrant == "ne":
            top_pad, left_pad, bottom_pad, right_pad = small(), large(), large(), small()
        elif quadrant == "sw":
            top_pad, left_pad, bottom_pad, right_pad = large(), small(), small(), large()
        else:
            top_pad, left_pad, bottom_pad, right_pad = large(), large(), small(), small()

        inner_h = child_h + top_pad + bottom_pad
        inner_w = child_w + left_pad + right_pad
        if inner_h < pat_h + 1 or inner_w < pat_w + 1:
            continue

        pat_rect = ((inner_h - pat_h) // 2, (inner_w - pat_w) // 2, pat_h, pat_w)
        child_rect = (top_pad, left_pad, child_h, child_w)
        if _rects_intersect(child_rect, pat_rect, margin=1):
            continue

        dr = 1 + top_pad
        dc = 1 + left_pad
        frames = [{"bbox": (0, 0, inner_h + 1, inner_w + 1), "orientation": int(orientation)}]
        for frame in structure["frames"]:
            r0, c0, r1, c1 = frame["bbox"]
            frames.append(
                {
                    "bbox": (int(r0) + dr, int(c0) + dc, int(r1) + dr, int(c1) + dc),
                    "orientation": int(frame["orientation"]),
                }
            )
        return {"height": inner_h + 2, "width": inner_w + 2, "frames": frames}

    return None


def _compose_widths(total: int, parts: int, min_each: int) -> list[int] | None:
    if total < parts * min_each:
        return None
    widths: list[int] = []
    remaining = int(total)
    for idx in range(parts):
        if idx == parts - 1:
            widths.append(remaining)
            break
        min_rest = (parts - idx - 1) * min_each
        value = random.randint(min_each, remaining - min_rest)
        widths.append(value)
        remaining -= value
    random.shuffle(widths)
    return widths


def _render_input(
    structure: dict,
    pattern: np.ndarray,
    palette: list[int],
    top_margin: int,
    left_margin: int,
    right_margin: int,
    bottom_gap: int,
) -> np.ndarray | None:
    struct_h = int(structure["height"])
    struct_w = int(structure["width"])
    grid_h = top_margin + struct_h + bottom_gap + 1
    grid_w = left_margin + struct_w + right_margin
    if not (10 <= grid_h <= 30 and 10 <= grid_w <= 30):
        return None

    band_widths = _compose_widths(grid_w, len(palette), 2)
    if band_widths is None:
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    for frame in structure["frames"]:
        r0, c0, r1, c1 = frame["bbox"]
        shifted = (r0 + top_margin, c0 + left_margin, r1 + top_margin, c1 + left_margin)
        _draw_border(grid, shifted, ORANGE)
        mr, mc = _corner_cell(shifted, int(frame["orientation"]))
        grid[mr, mc] = RED

    inner = structure["frames"][-1]
    r0, c0, r1, c1 = inner["bbox"]
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    off_r, off_c = _center_offset(inner_h, inner_w, pattern.shape)
    top = top_margin + r0 + 1 + off_r
    left = left_margin + c0 + 1 + off_c
    patch = grid[top : top + pattern.shape[0], left : left + pattern.shape[1]]
    patch[pattern] = GREY

    start = 0
    for color, width in zip(palette, band_widths):
        grid[-1, start : start + width] = int(color)
        start += width
    assert start == grid_w
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_frames = 2
    palette = random.sample(PALETTE_POOL, n_frames)
    pattern = random.choice(PATTERNS)
    orientations = [random.randint(0, 3) for _ in range(n_frames)]

    structure = _inner_structure(pattern, orientations[-1])
    for orientation in reversed(orientations[:-1]):
        wrapped = _wrap_structure(structure, pattern, orientation)
        if wrapped is None:
            return None
        structure = wrapped

    top_margin = random.randint(0, 3)
    left_margin = random.randint(0, 3)
    right_margin = random.randint(0, 3)
    bottom_gap = random.randint(1, 3)
    grid = _render_input(structure, pattern, palette, top_margin, left_margin, right_margin, bottom_gap)
    if grid is None:
        return None

    try:
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
