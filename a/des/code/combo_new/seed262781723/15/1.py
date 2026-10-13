from __future__ import annotations

from collections import deque
import random

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
TEMPLATE_COLOR = 8
CORNER_NAMES = ("TL", "TR", "BL", "BR")
SEED_COLORS = (1, 2, 3, 4, 6, 7, 9)
MAX_GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for pts in _components(grid == FRAME_COLOR):
        r0, c0, r1, c1 = _bbox(pts)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(sub, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(sub[border] == FRAME_COLOR):
            continue
        if np.any(sub[1:-1, 1:-1] == FRAME_COLOR):
            continue
        if int(np.count_nonzero(sub == FRAME_COLOR)) != int(border.sum()):
            continue
        frames.append((r0, c0, r1, c1))
    return frames


def _extract_templates(grid: np.ndarray) -> dict[str, np.ndarray]:
    h, w = grid.shape
    quadrants: dict[str, list[tuple[int, int]]] = {name: [] for name in CORNER_NAMES}
    for r, c in np.argwhere(grid == TEMPLATE_COLOR):
        if r < h / 2 and c < w / 2:
            quadrants["TL"].append((int(r), int(c)))
        elif r < h / 2 and c >= w / 2:
            quadrants["TR"].append((int(r), int(c)))
        elif r >= h / 2 and c < w / 2:
            quadrants["BL"].append((int(r), int(c)))
        else:
            quadrants["BR"].append((int(r), int(c)))

    templates: dict[str, np.ndarray] = {}
    for name, pts in quadrants.items():
        assert pts, f"missing {name} template"
        r0, c0, r1, c1 = _bbox(pts)
        templates[name] = grid[r0 : r1 + 1, c0 : c1 + 1] == TEMPLATE_COLOR
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each frame's corner seed as both a template selector and a recolor key.

    Input structure:
    - The background is black.
    - Several same-sized gray rectangular frames have one-cell-thick borders and black interiors.
    - Each frame interior contains exactly one colored seed pixel in one of the four interior corners.
    - Four azure template patterns sit near the four outer grid corners; each template has the same size as a frame interior.

    Rule:
    - Detect every gray frame and read which interior corner contains its seed pixel.
    - Select the azure template from the matching outer corner region: top-left, top-right, bottom-left, or bottom-right.
    - Repaint that template inside the frame using the seed color, leaving the template's black cells black.
    - Remove the original seed pixel by replacing the whole frame interior with the recolored template.

    Key insight:
    - The seed pixel carries two independent signals at once: its corner position chooses one of four edge templates, and its color specifies how to recolor the chosen template.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _find_frames(g)
    assert frames, "expected at least one gray frame"

    interior_shapes = {(r1 - r0 - 1, c1 - c0 - 1) for r0, c0, r1, c1 in frames}
    assert len(interior_shapes) == 1, "all frame interiors must share one size"
    inner_h, inner_w = interior_shapes.pop()

    templates = _extract_templates(g)
    for name in CORNER_NAMES:
        assert templates[name].shape == (inner_h, inner_w), f"{name} template size must match frame interior"

    corner_lookup = {
        (0, 0): "TL",
        (0, inner_w - 1): "TR",
        (inner_h - 1, 0): "BL",
        (inner_h - 1, inner_w - 1): "BR",
    }

    for r0, c0, r1, c1 in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_pts = np.argwhere((inner != BG_COLOR) & (inner != FRAME_COLOR) & (inner != TEMPLATE_COLOR))
        assert len(seed_pts) == 1, "each frame interior must contain exactly one seed"
        sr, sc = map(int, seed_pts[0])
        assert (sr, sc) in corner_lookup, "seed must be placed on an interior corner"
        template_name = corner_lookup[(sr, sc)]
        seed_color = int(inner[sr, sc])
        out[r0 + 1 : r1, c0 + 1 : c1] = np.where(templates[template_name], seed_color, BG_COLOR)

    return out


def _manhattan_path(a: tuple[int, int], b: tuple[int, int], rng: random.Random) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    path = [(r, c)]
    while (r, c) != (tr, tc):
        steps = []
        if r < tr:
            steps.append((r + 1, c))
        elif r > tr:
            steps.append((r - 1, c))
        if c < tc:
            steps.append((r, c + 1))
        elif c > tc:
            steps.append((r, c - 1))
        r, c = rng.choice(steps)
        path.append((r, c))
    return path


def _random_template_mask(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(80):
        anchors = [
            (0, rng.randrange(w)),
            (h - 1, rng.randrange(w)),
            (rng.randrange(h), 0),
            (rng.randrange(h), w - 1),
        ]
        order = anchors[:]
        rng.shuffle(order)
        cells = {order[0]}
        for nxt in order[1:]:
            cells.update(_manhattan_path(order[0], nxt, rng))
            order[0] = nxt
        if len(cells) >= h * w:
            continue

        target = rng.randint(max(4, len(cells)), h * w - 1)
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))

        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if rows != set(range(h)) or cols != set(range(w)):
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask
    return None


def _choose_layout(
    inner_h: int, inner_w: int, frame_count: int, rng: random.Random
) -> tuple[int, int, int, int, int, int, int, int] | None:
    frame_h, frame_w = inner_h + 2, inner_w + 2
    candidates = []
    for slot_rows, slot_cols in ((2, 2), (2, 3), (3, 2)):
        if slot_rows * slot_cols < frame_count:
            continue
        for vgap in (1, 2):
            for hgap in (1, 2):
                for top_gap in (1, 2, 3):
                    for bottom_gap in (1, 2, 3):
                        for left_gap in (1, 2, 3):
                            for right_gap in (1, 2, 3):
                                height = inner_h + top_gap + slot_rows * frame_h + (slot_rows - 1) * vgap + bottom_gap + inner_h
                                width = inner_w + left_gap + slot_cols * frame_w + (slot_cols - 1) * hgap + right_gap + inner_w
                                if height <= 30 and width <= 30:
                                    candidates.append(
                                        (slot_rows, slot_cols, vgap, hgap, top_gap, bottom_gap, left_gap, right_gap)
                                    )
    if not candidates:
        return None
    return rng.choice(candidates)


def _frame_seed_offset(name: str, inner_h: int, inner_w: int) -> tuple[int, int]:
    if name == "TL":
        return 0, 0
    if name == "TR":
        return 0, inner_w - 1
    if name == "BL":
        return inner_h - 1, 0
    return inner_h - 1, inner_w - 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    inner_h = rng.randint(3, 5)
    inner_w = rng.randint(3, 4)
    frame_count = rng.randint(4, 6)
    layout = _choose_layout(inner_h, inner_w, frame_count, rng)
    if layout is None:
        return None
    slot_rows, slot_cols, vgap, hgap, top_gap, bottom_gap, left_gap, right_gap = layout

    templates: dict[str, np.ndarray] = {}
    seen_masks: set[bytes] = set()
    for name in CORNER_NAMES:
        mask = _random_template_mask(inner_h, inner_w, rng)
        if mask is None:
            return None
        sig = mask.tobytes()
        if sig in seen_masks:
            return None
        seen_masks.add(sig)
        templates[name] = mask

    frame_h, frame_w = inner_h + 2, inner_w + 2
    height = inner_h + top_gap + slot_rows * frame_h + (slot_rows - 1) * vgap + bottom_gap + inner_h
    width = inner_w + left_gap + slot_cols * frame_w + (slot_cols - 1) * hgap + right_gap + inner_w
    grid = np.full((height, width), BG_COLOR, dtype=int)

    grid[:inner_h, :inner_w][templates["TL"]] = TEMPLATE_COLOR
    grid[:inner_h, width - inner_w :][templates["TR"]] = TEMPLATE_COLOR
    grid[height - inner_h :, :inner_w][templates["BL"]] = TEMPLATE_COLOR
    grid[height - inner_h :, width - inner_w :][templates["BR"]] = TEMPLATE_COLOR

    slot_positions = []
    base_r = inner_h + top_gap
    base_c = inner_w + left_gap
    for rr in range(slot_rows):
        for cc in range(slot_cols):
            slot_positions.append((base_r + rr * (frame_h + vgap), base_c + cc * (frame_w + hgap)))
    chosen_slots = rng.sample(slot_positions, frame_count)

    corner_labels = list(CORNER_NAMES)
    extra_count = frame_count - len(CORNER_NAMES)
    if extra_count > 0:
        corner_labels.extend(rng.sample(list(CORNER_NAMES), extra_count))
    rng.shuffle(corner_labels)
    seed_colors = rng.sample(list(SEED_COLORS), frame_count)

    for (r0, c0), corner_name, seed_color in zip(chosen_slots, corner_labels, seed_colors):
        r1 = r0 + frame_h - 1
        c1 = c0 + frame_w - 1
        grid[r0, c0 : c1 + 1] = FRAME_COLOR
        grid[r1, c0 : c1 + 1] = FRAME_COLOR
        grid[r0 : r1 + 1, c0] = FRAME_COLOR
        grid[r0 : r1 + 1, c1] = FRAME_COLOR

        sr, sc = _frame_seed_offset(corner_name, inner_h, inner_w)
        grid[r0 + 1 + sr, c0 + 1 + sc] = seed_color

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
