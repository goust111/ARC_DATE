from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, colors_present, find_components, get_bg, paste, random_free_location
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each framed motif by reflecting it to the opposite side of its frame.

    Input structure
    - A background canvas contains one or more large same-color rectangular frame
      objects and a smaller motif color inside each frame's bounding box.

    Rule
    - Infer the frame color as the unique non-background color with the largest
      area; the other non-background color is the motif color.
    - For each frame bounding box, reflect all motif pixels inside that box:
      wide frames mirror left-to-right, while tall frames mirror top-to-bottom.

    Key insight
    - The frame's aspect ratio identifies the symmetry direction, and the output is
      the union of the original motif with its reflection inside the same frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected exactly one frame color and one motif color"

    counts = {c: int(np.count_nonzero(g == c)) for c in colors}
    largest = max(counts.values())
    frame_candidates = [c for c, n in counts.items() if n == largest]
    assert len(frame_candidates) == 1, "frame color must be uniquely largest"
    frame_color = frame_candidates[0]

    motif_colors = [c for c in colors if c != frame_color]
    assert len(motif_colors) == 1, "expected one motif color"
    motif_color = motif_colors[0]

    frames = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        colors=[frame_color],
    )
    assert frames, "expected at least one frame object"

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        h = bbox_h(frame["bbox"])
        w = bbox_w(frame["bbox"])
        assert h != w, "frame bbox must have a clear orientation"

        box_mask = np.zeros(g.shape, dtype=bool)
        box_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        motif_coords = np.argwhere(box_mask & (g == motif_color))
        assert motif_coords.size > 0, "each frame must contain a motif"

        for r, c in motif_coords:
            if w > h:
                rr, cc = int(r), int(c0 + c1 - c)
            else:
                rr, cc = int(r0 + r1 - r), int(c)
            assert int(g[rr, cc]) != frame_color, "reflection must stay inside the frame interior"
            out[rr, cc] = motif_color

    return out

# --------------- generator (o5x) ---------------

# Canvas, colors, and placement bounds.
BG_COLOR = 0
COLOR_PALETTE = tuple(range(1, 10))
CANVAS_ROW_RANGE, CANVAS_COL_RANGE = (16, 28), (16, 26)
MIN_CANVAS_BY_FRAME_COUNT = {2: (18, 18), 3: (22, 21)}
PLACEMENT_PADDING, PLACEMENT_CONNECTIVITY = 1, 8

# Frame shape and dimension choices.
FRAME_COUNT_RANGE = (1, 3)
ORIENTATIONS = ("H", "V")
V_FRAME_HEIGHTS, V_FRAME_WIDTHS = (10, 12), (7, 9, 11)
H_FRAME_HEIGHTS, H_FRAME_WIDTHS = (7, 9), (10, 11)
CAP_HALF_SPAN, BAR_OFFSET, WALL_INSET = 1, 1, 2

# Motif counts and safe interior positions.
MOTIF_SIDES = ("near", "far")
MOTIF_PIXEL_RANGE = (4, 9)
MOTIF_SIDE_INSET, MOTIF_AXIS_MARGIN = 3, 2
SYMMETRIC_PAIR_MIN_OFFSET = 1

# Rejection sampling limit.
GEN_RETRIES = 300

FrameSpec = tuple[str, int, int, int]
PlacedFrame = tuple[str, int, int, int, int, int]


def _gen_v_frame(h: int, w: int, color: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    mid = w // 2
    cap = slice(mid - CAP_HALF_SPAN, mid + CAP_HALF_SPAN + 1)

    patch[0, cap] = color
    patch[BAR_OFFSET, :] = color
    patch[BAR_OFFSET, mid] = BG_COLOR
    patch[WALL_INSET : h - WALL_INSET, 0] = color
    patch[WALL_INSET : h - WALL_INSET, w - 1] = color
    patch[h - 1 - BAR_OFFSET, :] = color
    patch[h - 1 - BAR_OFFSET, mid] = BG_COLOR
    patch[h - 1, cap] = color
    return patch


def _gen_frame_patch(orient: str, h: int, w: int, color: int) -> np.ndarray:
    if orient == "V":
        return _gen_v_frame(h, w, color)
    return _gen_v_frame(w, h, color).T


def _gen_motif_orbits(
    orient: str, h: int, w: int, side: str
) -> tuple[list[tuple[tuple[int, int], ...]], list[tuple[tuple[int, int], ...]]]:
    singles: list[tuple[tuple[int, int], ...]] = []
    pairs: list[tuple[tuple[int, int], ...]] = []

    if orient == "V":
        rows = range(MOTIF_SIDE_INSET, h // 2) if side == "near" else range(h // 2, h - MOTIF_SIDE_INSET)
        center_c = w // 2
        safe_cols = set(range(MOTIF_AXIS_MARGIN, w - MOTIF_AXIS_MARGIN))
        for r in rows:
            singles.append(((r, center_c),))
            for d in range(SYMMETRIC_PAIR_MIN_OFFSET, w):
                c0, c1 = center_c - d, center_c + d
                if c0 in safe_cols and c1 in safe_cols:
                    pairs.append(((r, c0), (r, c1)))
        return singles, pairs

    cols = range(MOTIF_SIDE_INSET, w // 2) if side == "near" else range((w + 1) // 2, w - MOTIF_SIDE_INSET)
    center_r = h // 2
    safe_rows = set(range(MOTIF_AXIS_MARGIN, h - MOTIF_AXIS_MARGIN))
    for c in cols:
        singles.append(((center_r, c),))
        for d in range(SYMMETRIC_PAIR_MIN_OFFSET, h):
            r0, r1 = center_r - d, center_r + d
            if r0 in safe_rows and r1 in safe_rows:
                pairs.append(((r0, c), (r1, c)))
    return singles, pairs


def _gen_feasible_single_counts(n_single: int, n_pair: int, count: int) -> list[int]:
    return [s for s in range(1, min(n_single, count) + 1) if (count - s) % 2 == 0 and (count - s) // 2 <= n_pair]


def _gen_motif_coords(orient: str, h: int, w: int, count: int) -> list[tuple[int, int]]:
    side = random.choice(MOTIF_SIDES)
    singles, pairs = _gen_motif_orbits(orient, h, w, side)
    feasible = _gen_feasible_single_counts(len(singles), len(pairs), count)
    if not feasible:
        raise ValueError("motif count does not fit frame")

    n_single = random.choice(feasible)
    n_pair = (count - n_single) // 2
    chosen = random.sample(singles, n_single) + random.sample(pairs, n_pair)
    coords = [cell for orbit in chosen for cell in orbit]
    random.shuffle(coords)
    return coords


def _gen_reflect_coords(coords: list[tuple[int, int]], orient: str, h: int, w: int) -> list[tuple[int, int]]:
    if orient == "H":
        return [(r, w - 1 - c) for r, c in coords]
    return [(h - 1 - r, c) for r, c in coords]


def _gen_dim_options(orient: str, count: int) -> list[tuple[int, int]]:
    if orient == "V":
        dims = [(h, w) for h in V_FRAME_HEIGHTS for w in V_FRAME_WIDTHS]
    else:
        dims = [(h, w) for h in H_FRAME_HEIGHTS for w in H_FRAME_WIDTHS]

    out = []
    for h, w in dims:
        for side in MOTIF_SIDES:
            singles, pairs = _gen_motif_orbits(orient, h, w, side)
            if _gen_feasible_single_counts(len(singles), len(pairs), count):
                out.append((h, w))
                break
    return out


def _sample_frame_specs() -> list[FrameSpec] | None:
    """Sample frame orientation, dimensions, and motif count before placement."""
    frames = []
    for _ in range(random.randint(*FRAME_COUNT_RANGE)):
        orient = random.choice(ORIENTATIONS)
        count = random.randint(*MOTIF_PIXEL_RANGE)
        options = _gen_dim_options(orient, count)
        if not options:
            return None
        h, w = random.choice(options)
        frames.append((orient, h, w, count))
    return frames

def _place_frame_patches(
    out: np.ndarray, bbox_canvas: np.ndarray, frames: list[FrameSpec], frame_color: int
) -> list[PlacedFrame] | None:
    """Place non-overlapping frames and return specs with top-left positions."""
    placed = []
    for orient, h, w, count in sorted(frames, key=lambda item: item[1] * item[2], reverse=True):
        block = np.ones((h, w), dtype=int)
        try:
            r0, c0 = random_free_location(
                bbox_canvas, block, transparent=BG_COLOR,
                padding=PLACEMENT_PADDING, padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None
        paste(bbox_canvas, block, r0, c0, transparent=BG_COLOR)
        paste(out, _gen_frame_patch(orient, h, w, frame_color), r0, c0, transparent=BG_COLOR)
        placed.append((orient, h, w, count, r0, c0))
    return placed


def _paint_motifs(out: np.ndarray, placed: list[PlacedFrame], motif_color: int) -> list[tuple[int, int]]:
    """Paint symmetric motif pairs into output and return cells to erase for input."""
    erased: list[tuple[int, int]] = []
    for orient, h, w, count, r0, c0 in placed:
        motif = _gen_motif_coords(orient, h, w, count)
        reflected = _gen_reflect_coords(motif, orient, h, w)
        for r, c in motif + reflected:
            out[r0 + r, c0 + c] = motif_color
        erased.extend((r0 + r, c0 + c) for r, c in reflected)
    return erased

def _gen_attempt() -> np.ndarray | None:
    frame_color, motif_color = random.sample(COLOR_PALETTE, 2)
    frames = _sample_frame_specs()
    if frames is None:
        return None

    max_h = max(h for _orient, h, _w, _count in frames)
    max_w = max(w for _orient, _h, w, _count in frames)
    extra_h, extra_w = MIN_CANVAS_BY_FRAME_COUNT.get(len(frames), (0, 0))
    min_h = max(CANVAS_ROW_RANGE[0], max_h, extra_h)
    min_w = max(CANVAS_COL_RANGE[0], max_w, extra_w)
    if min_h > CANVAS_ROW_RANGE[1] or min_w > CANVAS_COL_RANGE[1]:
        return None

    H = random.randint(min_h, CANVAS_ROW_RANGE[1])
    W = random.randint(min_w, CANVAS_COL_RANGE[1])
    out = np.full((H, W), BG_COLOR, dtype=int)
    bbox_canvas = np.full((H, W), BG_COLOR, dtype=int)

    placed = _place_frame_patches(out, bbox_canvas, frames, frame_color)
    if placed is None:
        return None
    erased = _paint_motifs(out, placed, motif_color)

    inp = out.copy()
    for r, c in erased:
        inp[r, c] = BG_COLOR

    try:
        if not np.array_equal(main(inp), out):
            return None
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 18419cfa input")
