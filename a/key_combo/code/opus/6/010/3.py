import random
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from arc_utils import bbox_coords, find_components, get_bg


BG = 0
FRAME = 5
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (20, 24)
NUM_CUPS_CHOICES = (3, 4)
CAPACITY_CHOICES = (3, 4, 5, 6, 7)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _flood_cavity(local: np.ndarray, bg: int, frame_color: int) -> list[tuple[int, int]]:
    h, w = local.shape
    q: deque[tuple[int, int]] = deque()
    seen: set[tuple[int, int]] = set()
    for c in range(w):
        if int(local[0, c]) == int(bg):
            seen.add((0, c))
            q.append((0, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in seen or int(local[nr, nc]) != int(bg):
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return sorted(seen)


def _row_intervals(cavity: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    by_row: dict[int, list[int]] = {}
    for r, c in cavity:
        by_row.setdefault(int(r), []).append(int(c))
    intervals = []
    for r in sorted(by_row):
        cols = sorted(by_row[r])
        intervals.append((int(r), int(cols[0]), int(cols[-1])))
    return intervals


def _extract_cups(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    cups = []
    comps = find_components(
        grid,
        bg=bg,
        colors=[FRAME],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    for comp in comps:
        r0, c0, r1, c1 = bbox_coords(comp["coords"])
        local = grid[r0 : r1 + 1, c0 : c1 + 1]
        cavity_local = _flood_cavity(local, bg, FRAME)
        if not cavity_local:
            continue
        cavity_global = [(r0 + r, c0 + c) for r, c in cavity_local]
        intervals = _row_intervals(cavity_local)
        widths = [right - left + 1 for _, left, right in intervals]
        lefts = [left for _, left, _ in intervals]
        cups.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "capacity": int(len(cavity_global)),
                "cavity": cavity_global,
                "fill_order": sorted(cavity_global, key=lambda rc: (-rc[0], rc[1])),
                "irregular": len(set(widths)) > 1 or len(set(lefts)) > 1,
            }
        )
    cups.sort(key=lambda cup: cup["bbox"])
    return cups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match scattered marker counts to cup capacities, then fill each cup with its matched color.

    Input structure:
    - Black background with 3-4 separated grey cup frames, each an upward-open U or stepped-U cavity.
    - Each cup frame is a single 4-connected color-5 component whose interior cavity contains only background.
    - Outside the cups are scattered non-grey, non-black marker pixels in one distinct color per cup.

    Rule:
    - For each grey cup, compute the cavity cells reachable from the top opening inside that cup's bbox.
    - Count all marker pixels by color, ignoring the grey frames and black background.
    - Uniquely match each marker color to the cup whose cavity area equals that color's pixel count.
    - Erase the original markers, then fill the matched cup cavity bottom-to-top, left-to-right within each row, using the matched color.

    Key insight:
    - The decisive link is an equality between two independently derived quantities: a cup's geometric capacity and a marker color's global pixel count.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    cups = _extract_cups(g)
    capacities = [cup["capacity"] for cup in cups]
    assert len(capacities) == len(set(capacities)) and capacities, "cup capacities must be unique"

    marker_counts = Counter(int(v) for v in g.flatten() if int(v) not in {bg, FRAME})
    assert len(marker_counts) == len(cups), "expected one marker color per cup"
    assert sorted(marker_counts.values()) == sorted(capacities), "marker counts must match cup capacities"

    color_by_capacity: dict[int, int] = {}
    for color, count in marker_counts.items():
        assert int(count) not in color_by_capacity, "counts must identify cups uniquely"
        color_by_capacity[int(count)] = int(color)

    out = g.copy()
    out[(g != bg) & (g != FRAME)] = bg
    for cup in cups:
        fill_color = color_by_capacity[int(cup["capacity"])]
        for r, c in cup["fill_order"]:
            out[r, c] = fill_color
    return out


def _sample_widths(capacity: int, height: int) -> list[int]:
    widths = [1] * height
    remaining = int(capacity) - height
    while remaining > 0:
        choices = [i for i, width in enumerate(widths) if width < 4]
        if not choices:
            return []
        idx = random.choice(choices)
        widths[idx] += 1
        remaining -= 1
    random.shuffle(widths)
    return widths


def _build_cup_patch(capacity: int, *, force_irregular: bool) -> np.ndarray | None:
    for _ in range(120):
        height = random.randint(2, min(4, capacity))
        widths = _sample_widths(capacity, height)
        if not widths:
            continue
        patch_w = max(widths) + 2 + random.randint(0, 1)
        lefts: list[int] = []
        ok = True
        for row, width in enumerate(widths):
            max_left = patch_w - width - 1
            if max_left < 1:
                ok = False
                break
            if row == 0:
                candidates = list(range(1, max_left + 1))
            else:
                prev_left = lefts[-1]
                prev_right = prev_left + widths[row - 1] - 1
                candidates = []
                for left in range(1, max_left + 1):
                    right = left + width - 1
                    if max(left, prev_left) <= min(right, prev_right):
                        candidates.append(left)
            if not candidates:
                ok = False
                break
            lefts.append(random.choice(candidates))
        if not ok:
            continue

        irregular = len(set(widths)) > 1 or len(set(lefts)) > 1
        if force_irregular and not irregular:
            continue

        patch = np.full((height + 1, patch_w), FRAME, dtype=int)
        for r, (left, width) in enumerate(zip(lefts, widths)):
            patch[r, left : left + width] = BG

        if random.random() < 0.5:
            patch = np.fliplr(patch)
        return patch
    return None


def _boxes_clear(box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], gap: int = 2) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in others:
        if not (r1 + gap < or0 or or1 + gap < r0 or c1 + gap < oc0 or oc1 + gap < c0):
            return False
    return True


def _place_cups(grid: np.ndarray, patches: list[np.ndarray]) -> list[dict] | None:
    h, w = grid.shape
    order = sorted(range(len(patches)), key=lambda i: patches[i].size, reverse=True)
    placed: list[dict | None] = [None] * len(patches)
    boxes: list[tuple[int, int, int, int]] = []
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        placed_here = None
        for _ in range(150):
            r0 = random.randint(1, h - ph - 1)
            c0 = random.randint(1, w - pw - 1)
            box = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
            if not _boxes_clear(box, boxes, gap=2):
                continue
            placed_here = {"patch": patch, "bbox": box}
            boxes.append(box)
            break
        if placed_here is None:
            return None
        placed[idx] = placed_here
    assert all(item is not None for item in placed)
    return [item for item in placed if item is not None]


def _mark_neighbors(mask: np.ndarray, r: int, c: int) -> None:
    h, w = mask.shape
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            mask[rr, cc] = True


def _place_markers(
    grid: np.ndarray,
    counts_by_color: dict[int, int],
    cup_boxes: list[tuple[int, int, int, int]],
) -> dict[int, list[tuple[int, int]]] | None:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    for r0, c0, r1, c1 in cup_boxes:
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h - 1, r1 + 1)
        cc1 = min(w - 1, c1 + 1)
        blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

    markers: dict[int, list[tuple[int, int]]] = {}
    for color, count in sorted(counts_by_color.items(), key=lambda item: -item[1]):
        cells: list[tuple[int, int]] = []
        for _ in range(count):
            candidates = [tuple(rc) for rc in np.argwhere(~blocked)]
            if not candidates:
                return None
            r, c = random.choice(candidates)
            r = int(r)
            c = int(c)
            cells.append((r, c))
            blocked[r, c] = True
            _mark_neighbors(blocked, r, c)
        markers[int(color)] = cells
    return markers


def _render_instance(
    shape: tuple[int, int],
    placements: list[dict],
    counts_by_color: dict[int, int],
    markers: dict[int, list[tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BG, dtype=int)
    expected = np.full(shape, BG, dtype=int)
    color_by_capacity = {count: color for color, count in counts_by_color.items()}

    for placed in placements:
        patch = placed["patch"]
        r0, c0, r1, c1 = placed["bbox"]
        inp[r0 : r1 + 1, c0 : c1 + 1] = patch
        expected[r0 : r1 + 1, c0 : c1 + 1] = patch

        cavity_local = _flood_cavity(patch, BG, FRAME)
        fill_color = int(color_by_capacity[len(cavity_local)])
        for rr, cc in cavity_local:
            expected[r0 + rr, c0 + cc] = fill_color

    for color, cells in markers.items():
        for r, c in cells:
            inp[r, c] = int(color)

    return inp, expected


def _validate_generated(
    inp: np.ndarray,
    expected: np.ndarray,
    capacities: list[int],
    placements: list[dict],
    markers: dict[int, list[tuple[int, int]]],
) -> bool:
    if np.array_equal(inp, expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    recovered = _extract_cups(inp)
    if len(recovered) != len(capacities):
        return False
    if sorted(cup["capacity"] for cup in recovered) != sorted(capacities):
        return False
    if len(set(capacities)) != len(capacities):
        return False
    if not any(cup["irregular"] for cup in recovered):
        return False

    marker_counts = {int(color): len(cells) for color, cells in markers.items()}
    if sorted(marker_counts.values()) != sorted(capacities):
        return False

    boxes = [placed["bbox"] for placed in placements]
    for i, box in enumerate(boxes):
        if not _boxes_clear(box, boxes[:i] + boxes[i + 1 :], gap=2):
            return False
    for cells in markers.values():
        for r, c in cells:
            if int(inp[r, c]) == FRAME:
                return False
            for r0, c0, r1, c1 in boxes:
                if r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1:
                    return False
    if np.count_nonzero((inp != BG) & (inp != FRAME)) != sum(capacities):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    num_cups = random.choice(NUM_CUPS_CHOICES)
    capacities = random.sample(CAPACITY_CHOICES, k=num_cups)
    colors = random.sample(MARKER_COLORS, k=num_cups)
    counts_by_color = {int(color): int(capacity) for color, capacity in zip(colors, capacities)}

    forced_irregular = random.randrange(num_cups)
    patches = []
    for idx, capacity in enumerate(capacities):
        patch = _build_cup_patch(capacity, force_irregular=(idx == forced_irregular or random.random() < 0.35))
        if patch is None:
            return None
        patches.append(patch)

    placements = _place_cups(np.full((h, w), BG, dtype=int), patches)
    if placements is None:
        return None

    cup_boxes = [placed["bbox"] for placed in placements]
    markers = _place_markers(np.full((h, w), BG, dtype=int), counts_by_color, cup_boxes)
    if markers is None:
        return None

    inp, expected = _render_instance((h, w), placements, counts_by_color, markers)
    if not _validate_generated(inp, expected, capacities, placements, markers):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid cup-count matching input")
