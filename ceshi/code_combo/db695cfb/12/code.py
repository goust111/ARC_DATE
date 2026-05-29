from __future__ import annotations

# Combo variant of db695cfb (index 12)
# combo_spec: D14/2+D15/0
# Name: 区域填色+交叉触发新延伸

"""
Design analysis for the fused variant.

(a) Concrete behavior:
- Original: blue endpoints define finite diagonal segments; magenta pixels lying on those
  segments emit one perpendicular diagonal ray each; drawing stops there.
- Fused: the first magenta rays still start from trigger pixels on the blue segments, but
  every crossing between the two diagonal orientations becomes a fresh seed for the opposite
  orientation. Example: with three parallel blue "\" segments and two magenta "/" triggers
  on the middle segment, the two initial "/" rays hit the untouched outer blue segments,
  those hits extend the outer "\" diagonals to the border, and the enlarged "\" lines then
  keep confirming more "/" crossings. The final line family forms a diagonal lattice.
- After propagation, the lattice is treated as walls. Background components touching the
  frame stay background; enclosed components are map-colored with two alternating fill colors.

(b) Complexity calibration:
- The generator uses 3-5 parallel blue segments and 2-3 trigger pixels. This is enough to
  create repeated recursive crossings and multiple enclosed diamonds, but keeps the palette
  and object vocabulary small: background, blue structure, magenta triggers/rays, and two
  fill colors.

(c) Hidden assumptions broken from the original:
- Original code assumes crossings only matter through overwrite order. That no longer holds;
  orientation occupancy must be tracked separately from visible color.
- Original code assumes a trigger fires once from an input magenta pixel. That no longer
  holds; recursive seeds appear at line crossings and must be deduplicated.
- Original code keeps black/background untouched. That no longer holds; enclosed background
  components are recolored after all line growth finishes.

(d) Placement legality:
- All blue segments are parallel, interior, non-overlapping, and share parity so each active
  magenta line crosses every blue segment on integer cells.
- Trigger offsets are chosen away from segment endpoints so the initial magenta rays hit all
  blue segments inside their finite spans, guaranteeing legal recursive activation.

(e) Visual observability:
- The difference from the base task is visible in every valid sample: magenta lines do not
  stop at one hop, outer blue segments become border-to-border only after being hit, and the
  completed lattice creates filled diamond regions absent from the original task.

(f) Visual cleanliness:
- The generator avoids distractors and keeps one base orientation per sample. That prevents
  ambiguous blue pairings and reduces clutter at crossings.

(g) Inferability of color changes:
- Region colors are not assigned by hidden numeric ordering. They come from a two-color
  alternation on the enclosed-region adjacency graph, so neighboring enclosed cells visibly
  alternate in a checkerboard-like way across the diagonal lattice.
"""

from collections import deque
import random

import numpy as np

from arc_utils import get_bg, neighbors4


LINE_COLOR = 1
TRIGGER_COLOR = 6
FILL_COLORS = (2, 3)

ORIENT_A = 0  # "\" family
ORIENT_B = 1  # "/" family
ORIENT_DIRS = {
    ORIENT_A: ((1, 1), (-1, -1)),
    ORIENT_B: ((1, -1), (-1, 1)),
}


def _group_points(points: list[tuple[int, int]], orientation: int) -> dict[int, list[tuple[int, int]]]:
    groups: dict[int, list[tuple[int, int]]] = {}
    for r, c in points:
        key = r - c if orientation == ORIENT_A else r + c
        groups.setdefault(key, []).append((r, c))
    return groups


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    return [(r0 + k * dr, c0 + k * dc) for k in range(abs(r1 - r0) + 1)]


def _detect_base_orientation(line_points: list[tuple[int, int]]) -> int:
    best: tuple[int, int, int] | None = None
    for orientation in (ORIENT_A, ORIENT_B):
        groups = _group_points(line_points, orientation)
        for pts in groups.values():
            if len(pts) < 2:
                continue
            if orientation == ORIENT_A:
                pts = sorted(pts)
            else:
                pts = sorted(pts, key=lambda rc: (rc[0], -rc[1]))
            span = abs(pts[-1][0] - pts[0][0])
            cand = (span, len(pts), -orientation)
            if best is None or cand > best:
                best = cand
    assert best is not None, "at least one same-color diagonal pair is required"
    return -best[2]


def _component_labels(open_mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]], set[int]]:
    h, w = open_mask.shape
    labels = np.full((h, w), -1, dtype=int)
    components: list[list[tuple[int, int]]] = []
    border_ids: set[int] = set()

    for r in range(h):
        for c in range(w):
            if not open_mask[r, c] or labels[r, c] != -1:
                continue
            cid = len(components)
            q = deque([(r, c)])
            labels[r, c] = cid
            cells: list[tuple[int, int]] = []
            touches_border = False
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_border = True
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and open_mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = cid
                        q.append((nr, nc))
            components.append(cells)
            if touches_border:
                border_ids.add(cid)

    return labels, components, border_ids


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    line_points = [(int(r), int(c)) for r, c in zip(*np.where(g == LINE_COLOR))]
    assert line_points, "expected blue endpoints"

    base_orientation = _detect_base_orientation(line_points)
    perp_orientation = ORIENT_B if base_orientation == ORIENT_A else ORIENT_A

    groups = _group_points(line_points, base_orientation)
    segments: list[list[tuple[int, int]]] = []
    for pts in groups.values():
        if len(pts) < 2:
            continue
        if base_orientation == ORIENT_A:
            pts = sorted(pts)
        else:
            pts = sorted(pts, key=lambda rc: (rc[0], -rc[1]))
        segments.append(_diag_path(pts[0], pts[-1]))

    assert segments, "expected at least one drawable blue segment"

    mask_a = np.zeros((h, w), dtype=bool)
    mask_b = np.zeros((h, w), dtype=bool)
    seed_seen: set[tuple[int, int, int]] = set()
    queue: deque[tuple[int, int, int]] = deque()
    initial_triggers = 0
    recursive_crossings = 0

    def mask_for(orientation: int) -> np.ndarray:
        return mask_a if orientation == ORIENT_A else mask_b

    def add_seed(r: int, c: int, orientation: int, *, recursive: bool) -> None:
        nonlocal initial_triggers, recursive_crossings
        key = (int(r), int(c), int(orientation))
        if key in seed_seen:
            return
        seed_seen.add(key)
        mask_for(orientation)[r, c] = True
        queue.append(key)
        if recursive:
            recursive_crossings += 1
        else:
            initial_triggers += 1

    for path in segments:
        for r, c in path:
            mask_for(base_orientation)[r, c] = True
            if int(g[r, c]) == TRIGGER_COLOR:
                add_seed(r, c, perp_orientation, recursive=False)

    while queue:
        r0, c0, orientation = queue.popleft()
        own = mask_for(orientation)
        other = mask_for(ORIENT_B if orientation == ORIENT_A else ORIENT_A)
        for dr, dc in ORIENT_DIRS[orientation]:
            r, c = r0, c0
            while True:
                r += dr
                c += dc
                if not (0 <= r < h and 0 <= c < w):
                    break
                own[r, c] = True
                if other[r, c]:
                    add_seed(r, c, ORIENT_B if orientation == ORIENT_A else ORIENT_A, recursive=True)

    walls = mask_a | mask_b
    open_mask = ~walls
    labels, components, border_ids = _component_labels(open_mask)
    adjacency = [set() for _ in components]

    for r in range(h):
        for c in range(w):
            if not walls[r, c]:
                continue
            neigh = {
                int(labels[nr, nc])
                for nr, nc in neighbors4(r, c)
                if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] != -1 and int(labels[nr, nc]) not in border_ids
            }
            neigh = list(neigh)
            for i in range(len(neigh)):
                for j in range(i + 1, len(neigh)):
                    a = neigh[i]
                    b = neigh[j]
                    adjacency[a].add(b)
                    adjacency[b].add(a)

    region_color = np.full((h, w), -1, dtype=int)
    comp_fill: dict[int, int] = {}
    enclosed = [cid for cid in range(len(components)) if cid not in border_ids]
    enclosed.sort(key=lambda cid: min(components[cid]))

    for cid in enclosed:
        if cid in comp_fill:
            continue
        comp_fill[cid] = FILL_COLORS[0]
        q = deque([cid])
        while q:
            cur = q.popleft()
            cur_col = comp_fill[cur]
            nxt_col = FILL_COLORS[1] if cur_col == FILL_COLORS[0] else FILL_COLORS[0]
            for nb in sorted(adjacency[cur]):
                if nb not in comp_fill:
                    comp_fill[nb] = nxt_col
                    q.append(nb)

    out = np.full((h, w), bg, dtype=int)
    out[mask_for(base_orientation)] = LINE_COLOR
    out[mask_for(perp_orientation)] = TRIGGER_COLOR
    for cid, color in comp_fill.items():
        for r, c in components[cid]:
            region_color[r, c] = color
    out[region_color != -1] = region_color[region_color != -1]

    metrics = {
        "initial_triggers": int(initial_triggers),
        "recursive_crossings": int(recursive_crossings),
        "enclosed_regions": int(len(enclosed)),
        "wall_cells": int(walls.sum()),
        "base_orientation": int(base_orientation),
    }
    return out, metrics


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a recursive diagonal lattice from blue endpoint pairs and then color the cells trapped inside it.

    Input structure:
    - The background is the most common color. Blue pixels are endpoints of one or more parallel diagonal segments.
    - Magenta pixels lie on those blue diagonals and act as initial activators for the perpendicular diagonal family.

    Rule:
    - First connect each blue endpoint pair along its shared diagonal.
    - Each magenta pixel emits a full perpendicular diagonal line. Whenever a line of one orientation crosses a line of the other orientation, the crossing becomes a fresh seed that extends the opposite orientation as well. Repeat until no new crossings create unseen extensions.
    - Treat the completed line network as walls: keep border-connected background unchanged, and fill every enclosed region with one of two alternating fill colors.

    Key insight:
    - The task is no longer a single-pass ray drawing problem. Crossings recursively activate more diagonals, and the resulting diagonal lattice must then be interpreted as a region partition before the final colors can be assigned.
    """

    out, _ = _simulate(grid)
    return out


# --------------- generator ---------------

_H_RANGE = (12, 20)
_W_RANGE = (12, 20)
_SEGMENT_COUNT = (3, 5)
_SEG_LEN_MIN = 6
_SEG_LEN_MARGIN = 2
_GEN_RETRIES = 400


def _safe_bg() -> int:
    return random.choice([0, 4, 5, 7, 8, 9])


def _mirror_if_needed(grid: np.ndarray) -> np.ndarray:
    return np.fliplr(grid) if random.random() < 0.5 else grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_H_RANGE)
    w = random.randint(*_W_RANGE)
    bg = _safe_bg()

    max_len = min(h, w) - 1 - _SEG_LEN_MARGIN
    if max_len < _SEG_LEN_MIN:
        return None

    n_segments = random.randint(*_SEGMENT_COUNT)
    step = random.choice([2, 2, 4])
    seg_len = random.randint(_SEG_LEN_MIN, max_len)

    c0_max = w - seg_len - 2 - step * (n_segments - 1)
    if c0_max < 1:
        return None
    r0_max = h - seg_len - 2
    if r0_max < 1:
        return None

    r0 = random.randint(1, r0_max)
    parity = random.choice([0, 1])
    c0_candidates = [c for c in range(1, c0_max + 1) if c % 2 == parity]
    if not c0_candidates:
        return None
    c0_base = random.choice(c0_candidates)
    c0_values = [c0_base + step * k for k in range(n_segments)]

    max_shift = step * (n_segments - 1) // 2
    if seg_len - max_shift < 2:
        return None
    trigger_offsets = list(range(max(2, max_shift), seg_len - max(1, max_shift) + 1))
    if len(trigger_offsets) < 2:
        return None

    trigger_count = min(random.randint(2, 3), len(trigger_offsets))
    chosen_offsets = sorted(random.sample(trigger_offsets, trigger_count))
    pivot = n_segments // 2

    inp = np.full((h, w), bg, dtype=int)
    for c0 in c0_values:
        inp[r0, c0] = LINE_COLOR
        inp[r0 + seg_len, c0 + seg_len] = LINE_COLOR
    for off in chosen_offsets:
        inp[r0 + off, c0_values[pivot] + off] = TRIGGER_COLOR

    inp = _mirror_if_needed(inp)

    try:
        _, metrics = _simulate(inp)
    except Exception:
        return None

    if metrics["initial_triggers"] < 2:
        return None
    if metrics["recursive_crossings"] < metrics["initial_triggers"]:
        return None
    if metrics["enclosed_regions"] < 2:
        return None
    if metrics["wall_cells"] <= seg_len * 2:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")


def fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    _, metrics = _simulate(grid)
    return metrics
