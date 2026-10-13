import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
RNG = np.random.default_rng()

GRID_HEIGHT_RANGE = (20, 30)
GRID_WIDTH_RANGE = (20, 30)
STAMP_COUNT_RANGE = (2, 4)
STAMP_SIDE_RANGE = (3, 5)
FRAME_INTERIOR_MIN = 3
FRAME_INTERIOR_MAX = 8
PLACEMENT_GAP = 1
GEN_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_border_mask(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    if int(mask.sum()) != 2 * h + 2 * w - 4:
        return False
    return bool(
        mask[0, :].all()
        and mask[-1, :].all()
        and mask[:, 0].all()
        and mask[:, -1].all()
        and not mask[1:-1, 1:-1].any()
    )


def _extract_patch(grid: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = grid[coords[:, 0], coords[:, 1]]
    return patch, (r0, c0, r1, c1)


def _find_frames(grid: np.ndarray) -> dict[int, dict]:
    frames: dict[int, dict] = {}

    for color in PALETTE:
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            if not _is_border_mask(mask):
                continue

            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0 or not np.all((inner == BLACK) | (inner == GREY)):
                continue

            anchors = np.argwhere(inner == GREY)
            if anchors.shape[0] != 1:
                continue

            assert color not in frames, "each frame color must be unique"
            ar, ac = map(int, anchors[0])
            frames[color] = {
                "bbox": (r0, c0, r1, c1),
                "anchor": (r0 + 1 + ar, c0 + 1 + ac),
            }

    return frames


def _find_stamps(grid: np.ndarray, frames: dict[int, dict]) -> dict[int, dict]:
    stamps: dict[int, dict] = {}
    search = np.array(grid, copy=True)
    for frame in frames.values():
        ar, ac = frame["anchor"]
        search[ar, ac] = BLACK

    for coords in _components(search != BLACK):
        vals = search[coords[:, 0], coords[:, 1]]
        colors = set(map(int, np.unique(vals)))
        if colors == {GREY}:
            continue
        if GREY not in colors or len(colors) != 2:
            continue

        main_color = next(color for color in colors if color != GREY)
        if main_color not in frames:
            continue

        grey_coords = coords[vals == GREY]
        if grey_coords.shape[0] != 1:
            continue

        patch, bbox = _extract_patch(search, coords)
        anchor = tuple(map(int, np.argwhere(patch == GREY)[0]))

        assert main_color not in stamps, "each stamp color must be unique"
        stamps[main_color] = {
            "patch": patch,
            "bbox": bbox,
            "coords": coords,
            "anchor": anchor,
        }

    return stamps


def _paint_stamp(
    out: np.ndarray,
    patch: np.ndarray,
    stamp_anchor: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    target_anchor: tuple[int, int],
    color: int,
) -> None:
    r0, c0, r1, c1 = frame_bbox
    top = r0 + 1
    left = c0 + 1
    bottom = r1 - 1
    right = c1 - 1
    ar, ac = stamp_anchor

    for pr, pc in np.argwhere(patch == color):
        dr = int(target_anchor[0] + int(pr) - ar)
        dc = int(target_anchor[1] + int(pc) - ac)
        if top <= dr <= bottom and left <= dc <= right:
            out[dr, dc] = color


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_frames(g)
    assert frames, "expected at least one target frame"
    stamps = _find_stamps(g, frames)
    assert set(stamps) == set(frames), "stamp colors and frame colors must match exactly"

    out = g.copy()
    for stamp in stamps.values():
        out[stamp["coords"][:, 0], stamp["coords"][:, 1]] = BLACK

    for color, stamp in stamps.items():
        frame = frames[color]
        _paint_stamp(
            out,
            stamp["patch"],
            stamp["anchor"],
            frame["bbox"],
            frame["anchor"],
            color,
        )

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each colored upper sprite into the same-colored lower frame using the shared grey anchor.

    Input structure:
    - The background is black.
    - The upper part contains several small connected stamps, each with one main color and exactly one grey anchor pixel.
    - The lower part contains one-pixel-thick rectangular frames whose border colors match the stamp colors.
    - Each frame interior is black except for one grey anchor pixel.

    Rule:
    - Match every stamp to the unique frame with the same non-grey color.
    - Translate the stamp so its grey anchor overlaps the grey anchor inside the matched frame.
    - Paint only the non-grey stamp pixels into the frame interior, clipping any part that would fall outside the interior, and leave the frame anchor grey.
    - Remove the original upper stamps while keeping all frames unchanged.

    Key insight:
    - Color determines which frame receives each stamp, while the duplicated grey pixel acts as a positioning anchor that also reveals the required clipping behavior.
    """
    return _solve(grid)


def _random_connected_mask(rng: np.random.Generator) -> np.ndarray:
    for _ in range(120):
        h = int(rng.integers(STAMP_SIDE_RANGE[0], STAMP_SIDE_RANGE[1] + 1))
        w = int(rng.integers(STAMP_SIDE_RANGE[0], STAMP_SIDE_RANGE[1] + 1))
        target_area = int(rng.integers(max(4, (h * w) // 2), h * w + 1))

        start = (int(rng.integers(h)), int(rng.integers(w)))
        cells = {start}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if _is_border_mask(mask):
            continue
        return mask

    raise RuntimeError("failed to build a stamp mask")


def _make_stamp(rng: np.random.Generator, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    for _ in range(120):
        mask = _random_connected_mask(rng)
        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = color

        boundary = np.argwhere(
            mask
            & (
                np.pad(~mask[1:, :], ((0, 1), (0, 0)), constant_values=True)
                | np.pad(~mask[:-1, :], ((1, 0), (0, 0)), constant_values=True)
                | np.pad(~mask[:, 1:], ((0, 0), (0, 1)), constant_values=True)
                | np.pad(~mask[:, :-1], ((0, 0), (1, 0)), constant_values=True)
            )
        )
        interior = np.argwhere(mask)
        anchor_pool = boundary if (boundary.size > 0 and rng.integers(2) == 0) else interior
        ar, ac = map(int, anchor_pool[int(rng.integers(len(anchor_pool)))])
        patch[ar, ac] = GREY

        k = int(rng.integers(4))
        if k:
            patch = np.rot90(patch, k)
        if rng.integers(2):
            patch = np.fliplr(patch)

        anchor = tuple(map(int, np.argwhere(patch == GREY)[0]))
        color_mask = patch == color
        if _is_border_mask(color_mask):
            continue
        return patch, anchor

    raise RuntimeError("failed to build a non-frame stamp")


def _make_frame(
    rng: np.random.Generator,
    color: int,
    stamp_patch: np.ndarray,
    stamp_anchor: tuple[int, int],
    mode: str,
) -> tuple[np.ndarray, tuple[int, int]]:
    sh, sw = map(int, stamp_patch.shape)
    sar, sac = stamp_anchor

    if mode == "fit":
        ih = int(rng.integers(sh, min(FRAME_INTERIOR_MAX, sh + 3) + 1))
        iw = int(rng.integers(sw, min(FRAME_INTERIOR_MAX, sw + 3) + 1))
        min_ar = sar
        max_ar = ih - 1 - (sh - 1 - sar)
        min_ac = sac
        max_ac = iw - 1 - (sw - 1 - sac)
        tar = int(rng.integers(min_ar, max_ar + 1))
        tac = int(rng.integers(min_ac, max_ac + 1))
    else:
        ih = int(rng.integers(FRAME_INTERIOR_MIN, FRAME_INTERIOR_MAX + 1))
        iw = int(rng.integers(FRAME_INTERIOR_MIN, FRAME_INTERIOR_MAX + 1))

        sides = []
        if sar > 0:
            sides.append("top")
        if sar < sh - 1:
            sides.append("bottom")
        if sac > 0:
            sides.append("left")
        if sac < sw - 1:
            sides.append("right")
        side = sides[int(rng.integers(len(sides)))]

        tar = int(rng.integers(ih))
        tac = int(rng.integers(iw))
        if side == "top":
            tar = 0
        elif side == "bottom":
            tar = ih - 1
        elif side == "left":
            tac = 0
        else:
            tac = iw - 1

    frame = np.zeros((ih + 2, iw + 2), dtype=int)
    frame[0, :] = color
    frame[-1, :] = color
    frame[:, 0] = color
    frame[:, -1] = color
    frame[1 + tar, 1 + tac] = GREY
    return frame, (tar, tac)


def _rect_clear(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, gap: int) -> bool:
    if r0 < 0 or c0 < 0 or r0 + h > occupied.shape[0] or c0 + w > occupied.shape[1]:
        return False
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(occupied.shape[0], r0 + h + gap)
    cc1 = min(occupied.shape[1], c0 + w + gap)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _reserve(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _place_in_band(
    rng: np.random.Generator,
    occupied: np.ndarray,
    band_top: int,
    band_bottom: int,
    h: int,
    w: int,
) -> tuple[int, int] | None:
    if band_bottom - band_top + 1 < h or occupied.shape[1] < w:
        return None

    row_lo = band_top
    row_hi = band_bottom - h + 1
    col_lo = 0
    col_hi = occupied.shape[1] - w

    for _ in range(200):
        r0 = int(rng.integers(row_lo, row_hi + 1))
        c0 = int(rng.integers(col_lo, col_hi + 1))
        if _rect_clear(occupied, r0, c0, h, w, PLACEMENT_GAP):
            _reserve(occupied, r0, c0, h, w)
            return r0, c0
    return None


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    target = grid[top : top + h, left : left + w]
    mask = patch != BLACK
    target[mask] = patch[mask]


def _render_expected(
    grid: np.ndarray,
    placements: list[dict],
    frame_specs: dict[int, dict],
) -> np.ndarray:
    out = grid.copy()
    for spec in placements:
        patch = spec["patch"]
        r0, c0 = spec["pos"]
        cells = np.argwhere(patch != BLACK)
        out[r0 + cells[:, 0], c0 + cells[:, 1]] = BLACK

    for color, spec in frame_specs.items():
        stamp = next(item for item in placements if item["color"] == color)
        _paint_stamp(
            out,
            stamp["patch"],
            stamp["anchor"],
            spec["bbox"],
            spec["anchor_abs"],
            color,
        )

    return out


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1] + 1))
    w = int(RNG.integers(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1] + 1))
    split = h // 2
    top_band = (1, split - 3)
    bottom_band = (split + 2, h - 2)
    if top_band[1] < top_band[0] or bottom_band[1] < bottom_band[0]:
        return None

    count = int(RNG.integers(STAMP_COUNT_RANGE[0], STAMP_COUNT_RANGE[1] + 1))
    colors = list(RNG.choice(PALETTE, size=count, replace=False))
    RNG.shuffle(colors)

    if count == 1:
        modes = ["fit"]
    else:
        clip_count = count // 2
        fit_count = count - clip_count
        modes = ["clip"] * clip_count + ["fit"] * fit_count
        RNG.shuffle(modes)

    stamps: list[dict] = []
    frame_inputs: list[dict] = []
    for color, mode in zip(colors, modes):
        patch, anchor = _make_stamp(RNG, int(color))
        frame, frame_anchor = _make_frame(RNG, int(color), patch, anchor, mode)
        stamps.append({"color": int(color), "patch": patch, "anchor": anchor})
        frame_inputs.append(
            {
                "color": int(color),
                "frame": frame,
                "anchor": frame_anchor,
            }
        )

    occupied = np.zeros((h, w), dtype=bool)
    grid = np.zeros((h, w), dtype=int)
    stamp_specs: list[dict] = []

    for stamp in sorted(stamps, key=lambda item: -item["patch"].size):
        sh, sw = stamp["patch"].shape
        pos = _place_in_band(RNG, occupied, top_band[0], top_band[1], sh, sw)
        if pos is None:
            return None
        _paste(grid, stamp["patch"], pos[0], pos[1])
        stamp_specs.append({**stamp, "pos": pos})

    frame_specs: dict[int, dict] = {}
    for spec in sorted(frame_inputs, key=lambda item: -item["frame"].size):
        fh, fw = spec["frame"].shape
        pos = _place_in_band(RNG, occupied, bottom_band[0], bottom_band[1], fh, fw)
        if pos is None:
            return None
        _paste(grid, spec["frame"], pos[0], pos[1])
        tar, tac = spec["anchor"]
        frame_specs[spec["color"]] = {
            "bbox": (pos[0], pos[1], pos[0] + fh - 1, pos[1] + fw - 1),
            "anchor_abs": (pos[0] + 1 + tar, pos[1] + 1 + tac),
        }

    expected = _render_expected(grid, stamp_specs, frame_specs)
    solved = _solve(grid)
    if not np.array_equal(solved, expected):
        return None
    if not np.any(expected != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
