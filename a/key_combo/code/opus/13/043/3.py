from __future__ import annotations

import numpy as np


FRAME_COLORS = (1, 3, 4, 6)
MASK_COLOR = 5
BACKGROUND = 0
DEPTH_TO_FILL = {0: 2, 1: 1, 2: 4}
FRAME_COUNT = 3
SIDES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in sorted(set(map(int, np.unique(grid))) - {BACKGROUND, MASK_COLOR}):
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 3:
            continue
        border = _border_mask(height, width)
        window = grid[r0 : r1 + 1, c0 : c1 + 1]
        if coords.shape[0] != int(border.sum()):
            continue
        if not np.all(window[border] == color):
            continue
        frames.append({"color": color, "bbox": bbox})

    assert len(frames) >= 3, "expected at least three rectangular frames"
    areas = [(f["bbox"][2] - f["bbox"][0] + 1) * (f["bbox"][3] - f["bbox"][1] + 1) for f in frames]
    for i, frame in enumerate(frames):
        bbox = frame["bbox"]
        parents = [j for j, other in enumerate(frames) if i != j and _strictly_contains(other["bbox"], bbox)]
        if not parents:
            frame["parent"] = None
            continue
        frame["parent"] = min(
            parents,
            key=lambda j: (areas[j], frames[j]["bbox"][0], frames[j]["bbox"][1]),
        )

    roots = [i for i, frame in enumerate(frames) if frame["parent"] is None]
    assert len(roots) == 1, "expected a single outermost frame"

    def depth_of(idx: int) -> int:
        parent = frames[idx]["parent"]
        if parent is None:
            return 0
        return depth_of(int(parent)) + 1

    for i, frame in enumerate(frames):
        frame["depth"] = depth_of(i)
        depth = int(frame["depth"])
        assert depth in DEPTH_TO_FILL, "encountered unsupported nesting depth"

    frames_by_old_index = list(frames)
    frames.sort(key=lambda f: int(f["depth"]))
    index_by_bbox = {tuple(frame["bbox"]): i for i, frame in enumerate(frames)}
    for frame in frames:
        old_parent = frame["parent"]
        if old_parent is None:
            frame["parent"] = None
        else:
            parent_bbox = tuple(frames_by_old_index[old_parent]["bbox"])
            frame["parent"] = index_by_bbox[parent_bbox]
        frame["children"] = []
    for i, frame in enumerate(frames):
        parent = frame["parent"]
        if parent is not None:
            frames[int(parent)]["children"].append(i)
    for frame in frames:
        assert len(frame["children"]) <= 1, "expected a single nesting chain"
        child_indices = frame["children"]
        frame["child_bbox"] = None if not child_indices else frames[int(child_indices[0])]["bbox"]
    return frames


def _mask_bbox_for_frame(grid: np.ndarray, frame: dict[str, object]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = frame["bbox"]
    annulus = np.zeros_like(grid, dtype=bool)
    annulus[r0 + 1 : r1, c0 + 1 : c1] = True
    child_bbox = frame["child_bbox"]
    if child_bbox is not None:
        cr0, cc0, cr1, cc1 = child_bbox
        annulus[cr0 : cr1 + 1, cc0 : cc1 + 1] = False
    coords = np.argwhere(annulus & (grid == MASK_COLOR))
    assert coords.size > 0, "expected a grey mask in each frame annulus"
    bbox = _bbox_coords(coords)
    mr0, mc0, mr1, mc1 = bbox
    region = grid[mr0 : mr1 + 1, mc0 : mc1 + 1]
    assert set(map(int, np.unique(region))).issubset({BACKGROUND, MASK_COLOR}), "mask region must be binary"
    assert np.any(region[0, :] == MASK_COLOR)
    assert np.any(region[-1, :] == MASK_COLOR)
    assert np.any(region[:, 0] == MASK_COLOR)
    assert np.any(region[:, -1] == MASK_COLOR)
    return bbox


def main(grid: np.ndarray) -> np.ndarray:
    """
    Invert each nested frame's grey mask inside its local bbox, using frame depth to choose the fill color.

    Input structure:
    - The grid has a black background with several one-cell-thick colored rectangular frames.
    - The frames form a strict containment chain, so each frame has a unique nesting depth.
    - Each frame owns one grey-on-black binary mask located in the part of its interior not occupied by its direct child frame.

    Rule:
    - Detect the rectangular frames and derive their nesting depths from spatial containment.
    - For each frame, find the bbox of its grey mask within that frame's own annulus.
    - Invert only that local bbox: grey cells become black, and black cells become a depth-coded fill color
      (depth 0 -> red 2, depth 1 -> blue 1, depth 2 -> yellow 4).
    - Preserve all frame borders and every cell outside those local mask bboxes.

    Key insight:
    - The grey masks alone do not determine the output color; the color comes from the frame's position in the nesting hierarchy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_frames(g)

    out = g.copy()
    for frame in frames:
        depth = int(frame["depth"])
        fill = DEPTH_TO_FILL[depth]
        mr0, mc0, mr1, mc1 = _mask_bbox_for_frame(g, frame)
        region = g[mr0 : mr1 + 1, mc0 : mc1 + 1]
        inverted = np.full(region.shape, fill, dtype=int)
        inverted[region == MASK_COLOR] = BACKGROUND
        out[mr0 : mr1 + 1, mc0 : mc1 + 1] = inverted

    return out


def _random_mask(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    area = height * width
    while True:
        mask = rng.integers(0, 2, size=(height, width), dtype=int).astype(bool)
        count = int(mask.sum())
        if count < 2 or count > area - 2:
            continue
        if not mask[0, :].any() or not mask[-1, :].any():
            continue
        if not mask[:, 0].any() or not mask[:, -1].any():
            continue
        return mask


def _distinct_masks(rng: np.random.Generator) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], bytes]] = set()
    while len(masks) < FRAME_COUNT:
        height = int(rng.integers(3, 5))
        width = int(rng.integers(3, 5))
        mask = _random_mask(rng, height, width)
        key = (mask.shape, mask.astype(np.uint8).tobytes())
        if key in seen:
            continue
        seen.add(key)
        masks.append(mask)
    return masks


def _parent_size(
    rng: np.random.Generator,
    child_shape: tuple[int, int],
    mask_shape: tuple[int, int],
    side: str,
) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    child_h, child_w = child_shape
    mask_h, mask_w = mask_shape
    mins = {"top": 2, "bottom": 2, "left": 2, "right": 2}
    if side in ("top", "bottom"):
        mins[side] = mask_h + 1
    else:
        mins[side] = mask_w + 1

    top = int(rng.integers(mins["top"], mins["top"] + 3))
    bottom = int(rng.integers(mins["bottom"], mins["bottom"] + 3))
    left = int(rng.integers(mins["left"], mins["left"] + 3))
    right = int(rng.integers(mins["right"], mins["right"] + 3))
    return (child_h + top + bottom, child_w + left + right), (top, left, bottom, right)


def _choose_mask_bbox(
    rng: np.random.Generator,
    frame_bbox: tuple[int, int, int, int],
    mask: np.ndarray,
    child_bbox: tuple[int, int, int, int] | None,
    side: str | None,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = frame_bbox
    mask_h, mask_w = mask.shape
    if child_bbox is None:
        rr = rng.integers(r0 + 1, r1 - mask_h + 1)
        cc = rng.integers(c0 + 1, c1 - mask_w + 1)
        return int(rr), int(cc), int(rr + mask_h - 1), int(cc + mask_w - 1)

    cr0, cc0, cr1, cc1 = child_bbox
    if side == "top":
        row_lo, row_hi = r0 + 1, cr0 - mask_h
        col_lo, col_hi = c0 + 1, c1 - mask_w
    elif side == "bottom":
        row_lo, row_hi = cr1 + 1, r1 - mask_h
        col_lo, col_hi = c0 + 1, c1 - mask_w
    elif side == "left":
        row_lo, row_hi = r0 + 1, r1 - mask_h
        col_lo, col_hi = c0 + 1, cc0 - mask_w
    else:
        assert side == "right"
        row_lo, row_hi = r0 + 1, r1 - mask_h
        col_lo, col_hi = cc1 + 1, c1 - mask_w

    assert row_lo <= row_hi and col_lo <= col_hi, "mask must fit inside the chosen annulus strip"
    rr = int(rng.integers(row_lo, row_hi + 1))
    cc = int(rng.integers(col_lo, col_hi + 1))
    return rr, cc, rr + mask_h - 1, cc + mask_w - 1


def _build_spec(rng: np.random.Generator) -> dict[str, object] | None:
    masks = _distinct_masks(rng)
    colors = [int(x) for x in rng.choice(FRAME_COLORS, size=FRAME_COUNT, replace=False)]
    side0 = str(rng.choice(SIDES))
    side1 = str(rng.choice(SIDES))
    sides = [side0, side1, None]

    inner_mask = masks[2]
    inner_h = int(rng.integers(max(5, inner_mask.shape[0] + 2), inner_mask.shape[0] + 6))
    inner_w = int(rng.integers(max(5, inner_mask.shape[1] + 2), inner_mask.shape[1] + 6))
    sizes = [None, None, (inner_h, inner_w)]
    margins: list[tuple[int, int, int, int] | None] = [None, None, None]

    size1, margins1 = _parent_size(rng, sizes[2], masks[1].shape, side1)
    sizes[1] = size1
    margins[1] = margins1
    size0, margins0 = _parent_size(rng, sizes[1], masks[0].shape, side0)
    sizes[0] = size0
    margins[0] = margins0

    outer_h, outer_w = sizes[0]
    edge_top = int(rng.integers(1, 4))
    edge_left = int(rng.integers(1, 4))
    edge_bottom = int(rng.integers(1, 4))
    edge_right = int(rng.integers(1, 4))
    grid_h = outer_h + edge_top + edge_bottom
    grid_w = outer_w + edge_left + edge_right
    if max(grid_h, grid_w) > 30:
        return None

    bboxes: list[tuple[int, int, int, int]] = [None, None, None]  # type: ignore[assignment]
    bboxes[0] = (edge_top, edge_left, edge_top + outer_h - 1, edge_left + outer_w - 1)
    for depth in (0, 1):
        top, left, _bottom, _right = margins[depth]
        child_h, child_w = sizes[depth + 1]
        pr0, pc0, _pr1, _pc1 = bboxes[depth]
        bboxes[depth + 1] = (
            pr0 + top,
            pc0 + left,
            pr0 + top + child_h - 1,
            pc0 + left + child_w - 1,
        )

    frames: list[dict[str, object]] = []
    for depth in range(FRAME_COUNT):
        child_bbox = bboxes[depth + 1] if depth + 1 < FRAME_COUNT else None
        mask_bbox = _choose_mask_bbox(rng, bboxes[depth], masks[depth], child_bbox, sides[depth])
        frames.append(
            {
                "depth": depth,
                "color": colors[depth],
                "bbox": bboxes[depth],
                "child_bbox": child_bbox,
                "mask": masks[depth],
                "mask_bbox": mask_bbox,
                "side": sides[depth],
            }
        )

    return {"shape": (grid_h, grid_w), "frames": frames}


def _render_input(spec: dict[str, object]) -> np.ndarray:
    grid = np.zeros(spec["shape"], dtype=int)
    for frame in spec["frames"]:
        color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color

    for frame in spec["frames"]:
        mask = frame["mask"]
        mr0, mc0, mr1, mc1 = frame["mask_bbox"]
        window = grid[mr0 : mr1 + 1, mc0 : mc1 + 1]
        window[mask] = MASK_COLOR
    return grid


def _render_expected(spec: dict[str, object], inp: np.ndarray) -> np.ndarray:
    out = inp.copy()
    for frame in spec["frames"]:
        fill = DEPTH_TO_FILL[int(frame["depth"])]
        mask = frame["mask"]
        mr0, mc0, mr1, mc1 = frame["mask_bbox"]
        region = out[mr0 : mr1 + 1, mc0 : mc1 + 1]
        region[:, :] = np.where(mask, BACKGROUND, fill)
    return out


def _validate_spec(spec: dict[str, object], inp: np.ndarray) -> bool:
    expected = _render_expected(spec, inp)
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    frames = _find_frames(inp)
    if len(frames) != FRAME_COUNT:
        return False

    expected_by_depth = {int(frame["depth"]): frame for frame in spec["frames"]}
    changed_union = np.zeros(inp.shape, dtype=bool)
    seen_masks: set[tuple[tuple[int, ...], bytes]] = set()

    for parsed in frames:
        depth = int(parsed["depth"])
        expected_frame = expected_by_depth[depth]
        if tuple(parsed["bbox"]) != tuple(expected_frame["bbox"]):
            return False
        if int(parsed["color"]) != int(expected_frame["color"]):
            return False

        mask = expected_frame["mask"]
        mask_key = (mask.shape, mask.astype(np.uint8).tobytes())
        if mask_key in seen_masks:
            return False
        seen_masks.add(mask_key)

        parsed_bbox = _mask_bbox_for_frame(inp, parsed)
        if tuple(parsed_bbox) != tuple(expected_frame["mask_bbox"]):
            return False

        mr0, mc0, mr1, mc1 = parsed_bbox
        region_in = inp[mr0 : mr1 + 1, mc0 : mc1 + 1]
        region_out = actual[mr0 : mr1 + 1, mc0 : mc1 + 1]
        fill = DEPTH_TO_FILL[depth]
        if not np.any(region_in == BACKGROUND):
            return False
        if not np.any(region_in == MASK_COLOR):
            return False
        if not np.array_equal(region_out, np.where(region_in == MASK_COLOR, BACKGROUND, fill)):
            return False
        changed_union[mr0 : mr1 + 1, mc0 : mc1 + 1] = True

    if np.any(actual[~changed_union] != inp[~changed_union]):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        spec = _build_spec(rng)
        if spec is None:
            continue
        inp = _render_input(spec)
        if _validate_spec(spec, inp):
            return inp
    raise RuntimeError("failed to generate a valid nested-depth inversion input")
