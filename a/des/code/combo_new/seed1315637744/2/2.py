from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 300
PACK_RETRIES = 150
CANVAS_LIMIT = 30


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _component_mask(coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _find_components(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            rows = [r0 for r0, _ in coords]
            cols = [c0 for _, c0 in coords]
            bbox = (min(rows), min(cols), max(rows), max(cols))
            comps.append({"coords": coords, "bbox": bbox, "area": len(coords)})
    return comps


def _is_rect_frame(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * (h + w) - 4
    if int(component["area"]) != expected:
        return False
    coord_set = set(component["coords"])
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in coord_set):
                return False
    return True


def _classify_side(frag_bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = frag_bbox
    fr0, fc0, fr1, fc1 = frame_bbox
    horizontal_overlap = not (c1 < fc0 or c0 > fc1)
    vertical_overlap = not (r1 < fr0 or r0 > fr1)

    if r1 < fr0 and horizontal_overlap:
        return "top"
    if r0 > fr1 and horizontal_overlap:
        return "bottom"
    if c1 < fc0 and vertical_overlap:
        return "left"
    if c0 > fc1 and vertical_overlap:
        return "right"

    frag_cr = (r0 + r1) / 2.0
    frag_cc = (c0 + c1) / 2.0
    frame_cr = (fr0 + fr1) / 2.0
    frame_cc = (fc0 + fc1) / 2.0
    dr = frag_cr - frame_cr
    dc = frag_cc - frame_cc
    if abs(dr) >= abs(dc):
        return "top" if dr < 0 else "bottom"
    return "left" if dc < 0 else "right"


def _paste_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    grid[top : top + h, left : left + w][mask] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each same-color fragment into its matching frame, docking it against the wall encoded by its source side.

    Input structure:
    - Black background with several one-pixel-thick hollow rectangular frames, one frame per foreground color.
    - Each frame interior is empty, and 1 to 3 small monochrome fragments of the same color sit outside that frame.
    - Every fragment is a separate 4-connected component and lies clearly above, below, left, or right of its matching frame.

    Rule:
    - For each color, identify the unique rectangular frame and all other same-color fragments outside it.
    - Erase every external fragment from its original position, then translate it into the frame interior without rotating or flipping it.
    - A fragment from above docks to the top inner wall; from below to the bottom inner wall; from the left to the left inner wall; from the right to the right inner wall.
    - If several fragments share one side, keep their original order along that side and pack them consecutively against the matching inner wall.

    Key insight:
    - Color selects the destination frame, while the fragment's relative side around that frame selects which interior wall it must hug.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for color in [int(v) for v in np.unique(g) if int(v) != bg]:
        comps = _find_components(g, color)
        frame_candidates = [comp for comp in comps if _is_rect_frame(comp)]
        if not frame_candidates:
            continue
        frame = max(frame_candidates, key=lambda comp: (int(comp["area"]), comp["bbox"]))
        fr0, fc0, fr1, fc1 = frame["bbox"]
        ir0, ic0 = fr0 + 1, fc0 + 1
        ir1, ic1 = fr1 - 1, fc1 - 1
        interior_h = ir1 - ir0 + 1
        interior_w = ic1 - ic0 + 1
        if interior_h <= 0 or interior_w <= 0:
            continue

        side_groups: dict[str, list[dict]] = defaultdict(list)
        for comp in comps:
            if comp is frame:
                continue
            for r, c in comp["coords"]:
                out[r, c] = bg
            r0, c0, r1, c1 = comp["bbox"]
            mask = _component_mask(comp["coords"], comp["bbox"])
            side = _classify_side(comp["bbox"], frame["bbox"])
            side_groups[side].append(
                {
                    "mask": mask,
                    "bbox": comp["bbox"],
                    "h": r1 - r0 + 1,
                    "w": c1 - c0 + 1,
                    "sort_key": (c0, r0) if side in ("top", "bottom") else (r0, c0),
                }
            )

        for side in SIDES:
            side_groups[side].sort(key=lambda item: item["sort_key"])

        top_band = max((item["h"] for item in side_groups["top"]), default=0)
        bottom_band = max((item["h"] for item in side_groups["bottom"]), default=0)
        left_band = max((item["w"] for item in side_groups["left"]), default=0)
        right_band = max((item["w"] for item in side_groups["right"]), default=0)

        middle_w = interior_w - left_band - right_band
        middle_h = interior_h - top_band - bottom_band
        if middle_w < 0 or middle_h < 0:
            continue
        if sum(item["w"] for item in side_groups["top"]) > middle_w:
            continue
        if sum(item["w"] for item in side_groups["bottom"]) > middle_w:
            continue
        if sum(item["h"] for item in side_groups["left"]) > middle_h:
            continue
        if sum(item["h"] for item in side_groups["right"]) > middle_h:
            continue

        x = ic0 + left_band
        for item in side_groups["top"]:
            _paste_mask(out, item["mask"], ir0, x, color)
            x += item["w"]

        x = ic0 + left_band
        for item in side_groups["bottom"]:
            _paste_mask(out, item["mask"], ir1 - item["h"] + 1, x, color)
            x += item["w"]

        y = ir0 + top_band
        for item in side_groups["left"]:
            _paste_mask(out, item["mask"], y, ic0, color)
            y += item["h"]

        y = ir0 + top_band
        for item in side_groups["right"]:
            _paste_mask(out, item["mask"], y, ic1 - item["w"] + 1, color)
            y += item["h"]

    return out


def _random_mask(rng: random.Random) -> np.ndarray:
    h = rng.randint(1, 3)
    w = rng.randint(1, 3)
    target = rng.randint(1, min(h * w, 4))
    seed = (rng.randrange(h), rng.randrange(w))
    cells = {seed}
    while len(cells) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _side_chain_length(items: list[dict], axis: str) -> int:
    if not items:
        return 0
    lengths = [int(item[axis]) for item in items]
    return sum(lengths) + len(lengths) - 1


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _make_cluster(rng: random.Random, color: int) -> np.ndarray | None:
    count = rng.randint(1, 3)
    fragments = []
    for _ in range(count):
        side = rng.choice(SIDES)
        mask = _random_mask(rng)
        fragments.append({"side": side, "mask": mask, "h": mask.shape[0], "w": mask.shape[1]})

    if count > 1 and len({frag["side"] for frag in fragments}) == 1 and rng.random() < 0.5:
        fragments[-1]["side"] = rng.choice([side for side in SIDES if side != fragments[0]["side"]])

    groups: dict[str, list[dict]] = defaultdict(list)
    for frag in fragments:
        groups[frag["side"]].append(frag)

    top_band = max((item["h"] for item in groups["top"]), default=0)
    bottom_band = max((item["h"] for item in groups["bottom"]), default=0)
    left_band = max((item["w"] for item in groups["left"]), default=0)
    right_band = max((item["w"] for item in groups["right"]), default=0)

    middle_w_needed = max(
        sum(item["w"] for item in groups["top"]),
        sum(item["w"] for item in groups["bottom"]),
        2,
    )
    middle_h_needed = max(
        sum(item["h"] for item in groups["left"]),
        sum(item["h"] for item in groups["right"]),
        2,
    )

    interior_w = left_band + middle_w_needed + right_band + rng.randint(0, 2)
    interior_h = top_band + middle_h_needed + bottom_band + rng.randint(0, 2)
    outer_w = interior_w + 2
    outer_h = interior_h + 2

    top_input_len = _side_chain_length(groups["top"], "w")
    bottom_input_len = _side_chain_length(groups["bottom"], "w")
    left_input_len = _side_chain_length(groups["left"], "h")
    right_input_len = _side_chain_length(groups["right"], "h")
    if top_input_len > outer_w or bottom_input_len > outer_w or left_input_len > outer_h or right_input_len > outer_h:
        return None

    gap_top = rng.randint(1, 2) if groups["top"] else 0
    gap_bottom = rng.randint(1, 2) if groups["bottom"] else 0
    gap_left = rng.randint(1, 2) if groups["left"] else 0
    gap_right = rng.randint(1, 2) if groups["right"] else 0

    top_extent = top_band + gap_top
    bottom_extent = bottom_band + gap_bottom
    left_extent = left_band + gap_left
    right_extent = right_band + gap_right

    grid = np.zeros((top_extent + outer_h + bottom_extent, left_extent + outer_w + right_extent), dtype=int)
    frame_bbox = (
        top_extent,
        left_extent,
        top_extent + outer_h - 1,
        left_extent + outer_w - 1,
    )
    _draw_frame(grid, frame_bbox, color)
    fr0, fc0, fr1, fc1 = frame_bbox

    if groups["top"]:
        x = fc0 + rng.randint(0, outer_w - top_input_len)
        for item in groups["top"]:
            y = fr0 - gap_top - item["h"]
            _paste_mask(grid, item["mask"], y, x, color)
            x += item["w"] + 1

    if groups["bottom"]:
        x = fc0 + rng.randint(0, outer_w - bottom_input_len)
        for item in groups["bottom"]:
            y = fr1 + gap_bottom + 1
            _paste_mask(grid, item["mask"], y, x, color)
            x += item["w"] + 1

    if groups["left"]:
        y = fr0 + rng.randint(0, outer_h - left_input_len)
        for item in groups["left"]:
            x = fc0 - gap_left - item["w"]
            _paste_mask(grid, item["mask"], y, x, color)
            y += item["h"] + 1

    if groups["right"]:
        y = fr0 + rng.randint(0, outer_h - right_input_len)
        for item in groups["right"]:
            x = fc1 + gap_right + 1
            _paste_mask(grid, item["mask"], y, x, color)
            y += item["h"] + 1

    return grid


def _pack_clusters(rng: random.Random, clusters: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.zeros((CANVAS_LIMIT, CANVAS_LIMIT), dtype=int)
    occupied = np.zeros_like(canvas, dtype=bool)

    for cluster in sorted(clusters, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
        ch, cw = cluster.shape
        placed = False
        for _ in range(PACK_RETRIES):
            top = rng.randint(0, CANVAS_LIMIT - ch)
            left = rng.randint(0, CANVAS_LIMIT - cw)
            r0 = max(0, top - 1)
            c0 = max(0, left - 1)
            r1 = min(CANVAS_LIMIT, top + ch + 1)
            c1 = min(CANVAS_LIMIT, left + cw + 1)
            if np.any(occupied[r0:r1, c0:c1]):
                continue
            canvas[top : top + ch, left : left + cw][cluster != 0] = cluster[cluster != 0]
            occupied[top : top + ch, left : left + cw] = True
            placed = True
            break
        if not placed:
            return None

    rows, cols = np.nonzero(canvas)
    if len(rows) == 0:
        return None
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    core = canvas[r0 : r1 + 1, c0 : c1 + 1]

    pad_top = rng.randint(0, min(2, CANVAS_LIMIT - core.shape[0]))
    pad_bottom = rng.randint(0, min(2, CANVAS_LIMIT - core.shape[0] - pad_top))
    pad_left = rng.randint(0, min(2, CANVAS_LIMIT - core.shape[1]))
    pad_right = rng.randint(0, min(2, CANVAS_LIMIT - core.shape[1] - pad_left))

    out = np.zeros((core.shape[0] + pad_top + pad_bottom, core.shape[1] + pad_left + pad_right), dtype=int)
    out[pad_top : pad_top + core.shape[0], pad_left : pad_left + core.shape[1]] = core
    return out


def _generator_sanity_check(grid: np.ndarray) -> bool:
    bg = _get_bg(grid)
    colors = [int(v) for v in np.unique(grid) if int(v) != bg]
    if not colors:
        return False
    for color in colors:
        comps = _find_components(grid, color)
        frames = [comp for comp in comps if _is_rect_frame(comp)]
        if len(frames) != 1:
            return False
        frame_bbox = frames[0]["bbox"]
        fragments = [comp for comp in comps if comp is not frames[0]]
        if not (1 <= len(fragments) <= 3):
            return False
        for comp in fragments:
            side = _classify_side(comp["bbox"], frame_bbox)
            if side not in SIDES:
                return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    color_count = rng.choice((2, 3))
    colors = rng.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], color_count)
    clusters: list[np.ndarray] = []
    for color in colors:
        cluster = None
        for _ in range(40):
            cluster = _make_cluster(rng, color)
            if cluster is not None and max(cluster.shape) <= 18:
                break
        if cluster is None:
            return None
        clusters.append(cluster)

    grid = _pack_clusters(rng, clusters)
    if grid is None or max(grid.shape) > CANVAS_LIMIT:
        return None
    if not _generator_sanity_check(grid):
        return None
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
