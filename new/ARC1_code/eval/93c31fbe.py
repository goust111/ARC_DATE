import numpy as np

from arc_utils import bbox_coords, colors_present, find_components, get_bg
import random
from itertools import product


BLUE = 1

_CORNER_ROLES = {
    ((0, 0), (0, 1), (1, 0)): "TL",
    ((0, 0), (0, 1), (1, 1)): "TR",
    ((0, 0), (1, 0), (1, 1)): "BL",
    ((0, 1), (1, 0), (1, 1)): "BR",
}


def _corner_key(role: str, box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if role == "TL":
        return r0, c0
    if role == "TR":
        return r0, c1
    if role == "BL":
        return r1, c0
    assert role == "BR"
    return r1, c1


def _find_frames(g: np.ndarray, marker: int) -> list[tuple[int, int, int, int]]:
    corners: dict[str, dict[tuple[int, int], None]] = {
        "TL": {},
        "TR": {},
        "BL": {},
        "BR": {},
    }

    comps = find_components(g, bg=get_bg(g), colors=[marker], connectivity=4)
    assert len(comps) > 0

    for comp in comps:
        coords = comp["coords"]
        assert int(comp["area"]) == 3
        r0, c0, r1, c1 = bbox_coords(coords)
        assert (r1 - r0, c1 - c0) == (1, 1)
        rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
        assert rel in _CORNER_ROLES
        role = _CORNER_ROLES[rel]
        key = _corner_key(role, (r0, c0, r1, c1))
        assert key not in corners[role]
        corners[role][key] = None

    frames = []
    used = []
    for top, left in corners["TL"]:
        for top2, right in corners["TR"]:
            if top2 != top or right <= left:
                continue
            for bottom, left2 in corners["BL"]:
                if left2 != left or bottom <= top:
                    continue
                if (bottom, right) in corners["BR"]:
                    frames.append((top, left, bottom, right))
                    used.extend(
                        [
                            ("TL", (top, left)),
                            ("TR", (top, right)),
                            ("BL", (bottom, left)),
                            ("BR", (bottom, right)),
                        ]
                    )

    total_corners = sum(len(by_role) for by_role in corners.values())
    assert len(used) == total_corners
    assert len(set(used)) == total_corners
    assert len(frames) > 0
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """Complete each colored corner frame by mirroring its internal blue pixels to the opposite side and remove blue pixels outside all frames.

    Input structure
    A black background contains blue pixels and one non-blue marker color. The marker
    color appears only as 2x2 L-corner triominoes, grouped into rectangular frames
    with exactly one top-left, top-right, bottom-left, and bottom-right corner.

    Rule
    Keep every non-blue pixel fixed. For each marker frame, keep the blue pixels
    inside its bounding rectangle and also paint their reflected copies: wide
    frames reflect left/right, while tall frames reflect top/bottom. Clear all
    blue pixels that are outside every marker frame.

    Key insight
    The colored L-corners define the frame bbox; the frame aspect ratio determines
    which pair of opposite sides the blue motif must be copied between.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    marker_colors = colors_present(g, exclude={bg, BLUE})
    assert len(marker_colors) == 1
    marker = int(marker_colors[0])
    frames = _find_frames(g, marker)

    out = g.copy()
    out[g == BLUE] = bg
    rr, cc = np.indices(g.shape)

    for top, left, bottom, right in frames:
        height = bottom - top + 1
        width = right - left + 1
        assert height != width

        in_frame = (top <= rr) & (rr <= bottom) & (left <= cc) & (cc <= right)
        blue_coords = np.argwhere((g == BLUE) & in_frame)
        assert len(blue_coords) > 0

        for r, c in blue_coords:
            r = int(r)
            c = int(c)
            if width > height:
                mr, mc = r, left + right - c
            else:
                mr, mc = top + bottom - r, c
            assert g[mr, mc] != marker
            out[r, c] = BLUE
            out[mr, mc] = BLUE

    return out

# --------------- generator (o5x) ---------------

# Canvas and palette choices.
GRID_ROWS_MIN, GRID_ROWS_MAX = 20, 28
GRID_COLS_MIN, GRID_COLS_MAX = 21, 28
MARKER_COLOR_MIN, MARKER_COLOR_MAX = 2, 8
GEN_BG = 0

# Frame layout bounds.
FRAME_COUNT_MIN, FRAME_COUNT_MAX = 2, 4
FRAME_SIZE_MIN, FRAME_SIZE_MAX = 5, 14
FRAME_BORDER_MARGIN = 1
FRAME_GAP_PAD = 1
FRAME_AREA_MIN, FRAME_AREA_MAX = 12, 70

# Motif density and clustering.
MOTIF_COUNT_MIN, MOTIF_COUNT_MAX = 2, 10
MOTIF_MIN_DENSITY_DENOM = 10
MOTIF_MAX_DENSITY_DENOM = 5
MOTIF_CLUSTER_PROB = 0.35
MOTIF_FRONTIER_PROB = 0.8
LOW_HALF_SIDE = 0
HIGH_HALF_SIDE = 1

# Distractor count and ratio controls.
DISTRACTOR_COUNT_MIN, DISTRACTOR_COUNT_MAX = 6, 13
DISTRACTOR_RATIO_MIN_PCT, DISTRACTOR_RATIO_MAX_PCT = 35, 47
DISTRACTOR_PAIR_PROB = 0.4
PERCENT_SCALE = 100

# Generation retry limit.
GEN_RETRY_LIMIT = 300

ORTHOGONAL_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DISTRACTOR_PAIR_DIRS = ((1, 0), (0, 1))


def _gen_padded_bounds(
    shape: tuple[int, int], r0: int, c0: int, r1: int, c1: int, pad: int
) -> tuple[int, int, int, int]:
    rows, cols = shape
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(rows - 1, r1 + pad),
        min(cols - 1, c1 + pad),
    )


def _gen_mark_frame(g: np.ndarray, box: tuple[int, int, int, int], marker: int) -> None:
    top, left, bottom, right = box
    g[top, left] = marker
    g[top, left + 1] = marker
    g[top + 1, left] = marker

    g[top, right - 1] = marker
    g[top, right] = marker
    g[top + 1, right] = marker

    g[bottom - 1, left] = marker
    g[bottom, left] = marker
    g[bottom, left + 1] = marker

    g[bottom - 1, right] = marker
    g[bottom, right - 1] = marker
    g[bottom, right] = marker


def _gen_box_is_clear(
    used: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = FRAME_GAP_PAD
) -> bool:
    rr0, cc0, rr1, cc1 = _gen_padded_bounds(used.shape, r0, c0, r1, c1, pad)
    return not bool(used[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _gen_mark_used(
    used: np.ndarray, box: tuple[int, int, int, int], pad: int = FRAME_GAP_PAD
) -> None:
    r0, c0, r1, c1 = box
    rr0, cc0, rr1, cc1 = _gen_padded_bounds(used.shape, r0, c0, r1, c1, pad)
    used[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _gen_count_bounds(box: tuple[int, int, int, int]) -> tuple[int, int] | None:
    top, left, bottom, right = box
    height = bottom - top + 1
    width = right - left + 1
    area = (height - 2) * (width - 2)
    lo = max(MOTIF_COUNT_MIN, (area + MOTIF_MIN_DENSITY_DENOM - 1) // MOTIF_MIN_DENSITY_DENOM)
    hi = min(MOTIF_COUNT_MAX, area // MOTIF_MAX_DENSITY_DENOM)
    if width > height:
        half_capacity = (height - 2) * ((width - 2) // 2)
    else:
        half_capacity = ((height - 2) // 2) * (width - 2)
    hi = min(hi, half_capacity)
    if lo > hi:
        return None
    return lo, hi


def _gen_frame_boxes(rows: int, cols: int, count: int) -> list[tuple[int, int, int, int]] | None:
    used = np.zeros((rows, cols), dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    used_tops: set[int] = set()
    used_bottoms: set[int] = set()
    used_lefts: set[int] = set()
    used_rights: set[int] = set()

    for _ in range(count):
        choices = []
        for height in range(FRAME_SIZE_MIN, FRAME_SIZE_MAX + 1):
            for width in range(FRAME_SIZE_MIN, FRAME_SIZE_MAX + 1):
                if height == width:
                    continue
                area = (height - 2) * (width - 2)
                if area < FRAME_AREA_MIN or area > FRAME_AREA_MAX:
                    continue
                if _gen_count_bounds((0, 0, height - 1, width - 1)) is None:
                    continue
                locs = []
                for top in range(FRAME_BORDER_MARGIN, rows - height):
                    bottom = top + height - 1
                    if top in used_tops or bottom in used_bottoms:
                        continue
                    for left in range(FRAME_BORDER_MARGIN, cols - width):
                        right = left + width - 1
                        if left in used_lefts or right in used_rights:
                            continue
                        if _gen_box_is_clear(used, top, left, bottom, right):
                            locs.append((top, left))
                if locs:
                    choices.append((height, width, locs))
        if not choices:
            return None

        height, width, locs = random.choice(choices)
        top, left = random.choice(locs)
        box = (top, left, top + height - 1, left + width - 1)
        boxes.append(box)
        _gen_mark_used(used, box)
        used_tops.add(box[0])
        used_lefts.add(box[1])
        used_bottoms.add(box[2])
        used_rights.add(box[3])

    return boxes


def _gen_choose_counts(
    boxes: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, ...], int] | None:
    options = []
    for box in boxes:
        bounds = _gen_count_bounds(box)
        if bounds is None:
            return None
        lo, hi = bounds
        options.append(range(lo, hi + 1))

    by_distractor_count: dict[int, list[tuple[int, ...]]] = {}
    for counts in product(*options):
        source_total = sum(counts)
        for distractors in range(DISTRACTOR_COUNT_MIN, DISTRACTOR_COUNT_MAX + 1):
            total = source_total + distractors
            if (
                DISTRACTOR_RATIO_MIN_PCT * total
                <= PERCENT_SCALE * distractors
                <= DISTRACTOR_RATIO_MAX_PCT * total
            ):
                by_distractor_count.setdefault(distractors, []).append(tuple(counts))

    if not by_distractor_count:
        return None

    distractors = random.choice(sorted(by_distractor_count))
    return random.choice(by_distractor_count[distractors]), distractors


def _gen_half_cells(box: tuple[int, int, int, int], side: int) -> list[tuple[int, int]]:
    top, left, bottom, right = box
    cells = []
    if right - left + 1 > bottom - top + 1:
        for r in range(top + 1, bottom):
            for c in range(left + 1, right):
                if (side == LOW_HALF_SIDE and 2 * c < left + right) or (
                    side == HIGH_HALF_SIDE and 2 * c > left + right
                ):
                    cells.append((r, c))
    else:
        for r in range(top + 1, bottom):
            for c in range(left + 1, right):
                if (side == LOW_HALF_SIDE and 2 * r < top + bottom) or (
                    side == HIGH_HALF_SIDE and 2 * r > top + bottom
                ):
                    cells.append((r, c))
    return cells


def _gen_pick_motif(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    if len(cells) < count:
        return None
    if count <= MOTIF_COUNT_MIN or random.random() < MOTIF_CLUSTER_PROB:
        return random.sample(cells, count)

    cell_set = set(cells)
    chosen = {random.choice(cells)}
    while len(chosen) < count:
        frontier = []
        for r, c in chosen:
            for dr, dc in ORTHOGONAL_DIRS:
                nbr = (r + dr, c + dc)
                if nbr in cell_set and nbr not in chosen:
                    frontier.append(nbr)
        if frontier and random.random() < MOTIF_FRONTIER_PROB:
            chosen.add(random.choice(frontier))
        else:
            remaining = [cell for cell in cells if cell not in chosen]
            chosen.add(random.choice(remaining))
    return list(chosen)


def _gen_mirror_cell(box: tuple[int, int, int, int], r: int, c: int) -> tuple[int, int]:
    top, left, bottom, right = box
    if right - left + 1 > bottom - top + 1:
        return r, left + right - c
    return top + bottom - r, c


def _gen_frame_mask(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for top, left, bottom, right in boxes:
        mask[top : bottom + 1, left : right + 1] = True
    return mask


def _gen_has_neighbor(g: np.ndarray, r: int, c: int, color: int = 1) -> bool:
    rr0, cc0, rr1, cc1 = _gen_padded_bounds(g.shape, r, c, r, c, 1)
    window = g[rr0 : rr1 + 1, cc0 : cc1 + 1] == color
    window[r - rr0, c - cc0] = False
    return bool(window.any())


def _gen_pair_is_clean(
    g: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int = 1
) -> bool:
    pair = {a, b}
    for r, c in pair:
        rr0, cc0, rr1, cc1 = _gen_padded_bounds(g.shape, r, c, r, c, 1)
        window = g[rr0 : rr1 + 1, cc0 : cc1 + 1] == color
        for pr, pc in pair:
            if rr0 <= pr <= rr1 and cc0 <= pc <= cc1:
                window[pr - rr0, pc - cc0] = False
        if window.any():
            return False
    return True


def _gen_add_distractors(
    inp: np.ndarray, frame_mask: np.ndarray, count: int, blue: int = 1
) -> bool:
    rows, cols = inp.shape
    outside = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if not frame_mask[r, c] and inp[r, c] == GEN_BG
    ]

    remaining = count
    while remaining > 0:
        if remaining >= 2 and random.random() < DISTRACTOR_PAIR_PROB:
            pairs = []
            for r, c in outside:
                if inp[r, c] != GEN_BG:
                    continue
                for dr, dc in DISTRACTOR_PAIR_DIRS:
                    rr = r + dr
                    cc = c + dc
                    if (
                        rr < rows
                        and cc < cols
                        and not frame_mask[rr, cc]
                        and inp[rr, cc] == GEN_BG
                        and _gen_pair_is_clean(inp, (r, c), (rr, cc), blue)
                    ):
                        pairs.append(((r, c), (rr, cc)))
            if pairs:
                a, b = random.choice(pairs)
                inp[a] = blue
                inp[b] = blue
                remaining -= 2
                continue

        singles = [
            (r, c)
            for r, c in outside
            if inp[r, c] == GEN_BG and not _gen_has_neighbor(inp, r, c, blue)
        ]
        if not singles:
            singles = [(r, c) for r, c in outside if inp[r, c] == GEN_BG]
        if not singles:
            return False
        r, c = random.choice(singles)
        inp[r, c] = blue
        remaining -= 1
    return True


def _gen_layout_plan() -> tuple[int, int, int, list[tuple[int, int, int, int]], tuple[int, ...], int] | None:
    rows = random.randint(GRID_ROWS_MIN, GRID_ROWS_MAX)
    cols = random.randint(GRID_COLS_MIN, GRID_COLS_MAX)
    marker = random.randint(MARKER_COLOR_MIN, MARKER_COLOR_MAX)
    frame_count = random.randint(FRAME_COUNT_MIN, FRAME_COUNT_MAX)
    boxes = _gen_frame_boxes(rows, cols, frame_count)
    if boxes is None:
        return None
    counts_and_distractors = _gen_choose_counts(boxes)
    if counts_and_distractors is None:
        return None
    motif_counts, distractor_count = counts_and_distractors
    return rows, cols, marker, boxes, motif_counts, distractor_count


def _gen_place_frame_motifs(
    inp: np.ndarray,
    out: np.ndarray,
    marker: int,
    boxes: list[tuple[int, int, int, int]],
    motif_counts: tuple[int, ...],
) -> bool:
    for box, motif_count in zip(boxes, motif_counts):
        side = random.randint(LOW_HALF_SIDE, HIGH_HALF_SIDE)
        cells = _gen_half_cells(box, side)
        motif = _gen_pick_motif(cells, motif_count)
        if motif is None:
            return False
        for r, c in motif:
            mr, mc = _gen_mirror_cell(box, r, c)
            if out[mr, mc] == marker:
                return False
            out[r, c] = BLUE
            out[mr, mc] = BLUE
            inp[r, c] = BLUE
    return True


def _gen_attempt() -> np.ndarray | None:
    plan = _gen_layout_plan()
    if plan is None:
        return None
    rows, cols, marker, boxes, motif_counts, distractor_count = plan

    out = np.zeros((rows, cols), dtype=int)
    for box in boxes:
        _gen_mark_frame(out, box, marker)
    inp = out.copy()
    if not _gen_place_frame_motifs(inp, out, marker, boxes, motif_counts):
        return None
    frame_mask = _gen_frame_mask((rows, cols), boxes)
    if not _gen_add_distractors(inp, frame_mask, distractor_count):
        return None
    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 93c31fbe")
