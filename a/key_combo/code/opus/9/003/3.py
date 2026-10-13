from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from itertools import permutations
import random

import numpy as np


BLACK = 0
SIDE_TO_COLOR = {"top": 2, "bottom": 1, "left": 4, "right": 3}
SIDE_ORDER = ("top", "bottom", "left", "right")
PANEL_SIZE_RANGE = (6, 8)
PANEL_COLORS = [5, 6, 7, 8, 9]
ANCHOR_COLORS = [1, 2, 3, 4]
MAX_ATTEMPTS = 500


@dataclass(frozen=True)
class Panel:
    bbox: tuple[int, int, int, int]
    panel_color: int
    sprite_color: int
    anchor_color: int
    anchor: tuple[int, int]
    sprite_rel: tuple[tuple[int, int], ...]


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _connected_component(
    grid: np.ndarray,
    start: tuple[int, int],
    visited: set[tuple[int, int]],
    *,
    monochromatic: bool,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    sr, sc = start
    start_val = int(grid[sr, sc])
    q = deque([start])
    visited.add(start)
    out = []
    while q:
        r, c = q.popleft()
        out.append((r, c))
        for nr, nc in neighbors4(r, c):
            if not in_bounds((h, w), nr, nc) or (nr, nc) in visited or int(grid[nr, nc]) == BLACK:
                continue
            if monochromatic and int(grid[nr, nc]) != start_val:
                continue
            visited.add((nr, nc))
            q.append((nr, nc))
    return out


def find_components(grid: np.ndarray, *, monochromatic: bool) -> list[dict]:
    visited: set[tuple[int, int]] = set()
    comps = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) == BLACK or (r, c) in visited:
                continue
            coords = _connected_component(grid, (r, c), visited, monochromatic=monochromatic)
            comp_grid_vals = [int(grid[rr, cc]) for rr, cc in coords]
            color = comp_grid_vals[0] if monochromatic else None
            comps.append(
                {
                    "coords": coords,
                    "area": len(coords),
                    "bbox": bbox_coords(coords),
                    "color": color,
                }
            )
    return comps


def _is_connected(coords: list[tuple[int, int]]) -> bool:
    pts = set(coords)
    seen = {coords[0]}
    q = deque([coords[0]])
    while q:
        r, c = q.popleft()
        for nxt in neighbors4(r, c):
            if nxt in pts and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(pts)


def _parse_panel(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> Panel:
    r0, c0, r1, c1 = bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    vals, cnts = np.unique(patch, return_counts=True)
    counts = sorted(zip(cnts.tolist(), vals.tolist()), reverse=True)
    panel_color = int(counts[0][1])
    other = [int(v) for v in vals.tolist() if int(v) != panel_color]
    assert len(other) == 2, "panel must contain exactly one sprite color and one anchor color"

    by_count = sorted(((int(np.count_nonzero(patch == col)), col) for col in other))
    anchor_count, anchor_color = by_count[0]
    sprite_count, sprite_color = by_count[1]
    assert anchor_count == 1
    assert 3 <= sprite_count <= 5

    ar, ac = np.argwhere(patch == anchor_color)[0].tolist()
    anchor = (r0 + int(ar), c0 + int(ac))
    sprite_abs = [(r0 + int(rr), c0 + int(cc)) for rr, cc in np.argwhere(patch == sprite_color)]
    assert _is_connected(sprite_abs)
    sprite_rel = tuple(sorted((rr - anchor[0], cc - anchor[1]) for rr, cc in sprite_abs))
    return Panel(
        bbox=bbox,
        panel_color=panel_color,
        sprite_color=sprite_color,
        anchor_color=anchor_color,
        anchor=anchor,
        sprite_rel=sprite_rel,
    )


def _parse_grid(grid: np.ndarray) -> tuple[list[Panel], list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    panel_candidates = []
    markers = []
    for comp in find_components(g, monochromatic=False):
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if comp["area"] == h * w and PANEL_SIZE_RANGE[0] <= h <= PANEL_SIZE_RANGE[1] and PANEL_SIZE_RANGE[0] <= w <= PANEL_SIZE_RANGE[1]:
            panel_candidates.append(_parse_panel(g, comp["bbox"]))
        elif comp["area"] == 1:
            r, c = comp["coords"][0]
            markers.append((r, c, int(g[r, c])))
    panels = sorted(panel_candidates, key=lambda p: p.bbox[:2])
    assert 2 <= len(panels) <= 3
    return panels, markers


def _inside_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _contact_counts(copy_coords: list[tuple[int, int]], panels: list[Panel]) -> dict[str, int]:
    counts = {side: 0 for side in SIDE_ORDER}
    for r, c in copy_coords:
        for panel in panels:
            r0, c0, r1, c1 = panel.bbox
            if r == r0 - 1 and c0 <= c <= c1:
                counts["top"] += 1
            if r == r1 + 1 and c0 <= c <= c1:
                counts["bottom"] += 1
            if c == c0 - 1 and r0 <= r <= r1:
                counts["left"] += 1
            if c == c1 + 1 and r0 <= r <= r1:
                counts["right"] += 1
    return counts


def _winning_recolor(copy_coords: list[tuple[int, int]], sprite_color: int, panels: list[Panel]) -> tuple[int, dict[str, int]]:
    counts = _contact_counts(copy_coords, panels)
    best = max(counts.values())
    if best == 0:
        return sprite_color, counts
    side = next(side for side in SIDE_ORDER if counts[side] == best)
    return SIDE_TO_COLOR[side], counts


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    panels, markers = _parse_grid(g)
    anchor_to_idx = {panel.anchor_color: idx for idx, panel in enumerate(panels)}

    pending: dict[int, list[tuple[int, int]]] = {idx: [] for idx in range(len(panels))}
    matched_marker_colors = set()
    for r, c, color in markers:
        idx = anchor_to_idx.get(color)
        if idx is not None:
            pending[idx].append((r, c))
            matched_marker_colors.add(color)

    out = np.zeros_like(g)
    for panel in panels:
        r0, c0, r1, c1 = panel.bbox
        out[r0 : r1 + 1, c0 : c1 + 1] = panel.panel_color

    queue = [idx for idx in range(len(panels)) if pending[idx]]
    processed: set[int] = set()
    copies = []
    while queue:
        idx = queue.pop(0)
        if idx in processed:
            continue
        processed.add(idx)
        panel = panels[idx]
        seen_anchor_coords = set()
        for anchor_coord in pending[idx]:
            if anchor_coord in seen_anchor_coords:
                continue
            seen_anchor_coords.add(anchor_coord)
            copy_coords = [(anchor_coord[0] + dr, anchor_coord[1] + dc) for dr, dc in panel.sprite_rel]
            color, counts = _winning_recolor(copy_coords, panel.sprite_color, panels)
            for rr, cc in copy_coords:
                if in_bounds(g.shape, rr, cc):
                    out[rr, cc] = color
            copies.append(
                {
                    "panel_idx": idx,
                    "anchor": anchor_coord,
                    "coords": copy_coords,
                    "color": color,
                    "counts": counts,
                }
            )
            nxt = anchor_to_idx.get(color)
            if nxt is not None and nxt not in processed:
                pending[nxt].append(anchor_coord)
                if nxt not in queue:
                    queue.append(nxt)

    meta = {
        "panels": panels,
        "markers": markers,
        "copies": copies,
        "processed": processed,
        "matched_marker_colors": matched_marker_colors,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp panel sprites onto matching markers, recolor side-touching copies, and let matching recolors cascade into new stamps.

    Input structure:
    - The grid has a black background, 2-3 solid rectangular multicolor panels, and isolated single-pixel markers outside the panels.
    - Each panel is a filled 6x6 to 8x8 rectangle whose dominant color is the panel color, with one connected sprite in a second color and one single-pixel anchor in a third color.
    - Some outside markers share a color with a panel anchor, while other outside markers are unmatched noise.

    Rule:
    - For each panel that has matching outside markers, align the panel sprite to every marker using the panel anchor as the stamp anchor.
    - If a stamped copy touches the outside ring of any panel on the top, bottom, left, or right side, recolor the whole copy to the fixed side color (top=2, bottom=1, left=4, right=3); otherwise keep the sprite color.
    - A recolored copy whose new color matches another panel anchor activates that panel at the same anchor coordinate, producing a cascade; after the cascade ends, keep only the clean panel rectangles and all stamped copies, and erase all original interior symbols and all outside singleton markers.

    Key insight:
    - Recoloring is not terminal: the contacted side converts a stamped copy into a new color-key that can activate another panel, so the geometry of panel contact controls the chain reaction.
    """
    out, _ = _simulate(np.asarray(grid, dtype=int))
    return out


PATTERNS = [
    np.array(
        [
            [0, 9, 0],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 0],
            [1, 9, 0],
            [1, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 1, 9],
            [0, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 1, 9],
            [0, 1, 1],
        ],
        dtype=int,
    ),
]


def _transform_pattern(rng: random.Random) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    arr = PATTERNS[rng.randrange(len(PATTERNS))].copy()
    if rng.random() < 0.5:
        arr = np.fliplr(arr)
    arr = np.rot90(arr, rng.randrange(4))
    sprite = [(int(r), int(c)) for r, c in np.argwhere(arr == 1)]
    anchor = tuple(np.argwhere(arr == 9)[0].tolist())
    return sprite, anchor


def _place_panel_contents(
    rng: random.Random,
    bbox: tuple[int, int, int, int],
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    ph = r1 - r0 + 1
    pw = c1 - c0 + 1
    sprite_local, anchor_local = _transform_pattern(rng)
    max_r = max(r for r, _ in sprite_local + [anchor_local])
    max_c = max(c for _, c in sprite_local + [anchor_local])
    top = rng.randint(1, ph - max_r - 2)
    left = rng.randint(1, pw - max_c - 2)
    anchor = (r0 + top + anchor_local[0], c0 + left + anchor_local[1])
    sprite_abs = [(r0 + top + r, c0 + left + c) for r, c in sprite_local]
    return anchor, sprite_abs


def _bboxes_far_enough(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + margin < br0 or br1 + margin < ar0 or ac1 + margin < bc0 or bc1 + margin < ac0


def _sample_panel_bboxes(rng: random.Random, h: int, w: int, count: int) -> list[tuple[int, int, int, int]] | None:
    boxes = []
    for _ in range(300):
        ph = rng.randint(*PANEL_SIZE_RANGE)
        pw = rng.randint(*PANEL_SIZE_RANGE)
        r0 = rng.randint(1, h - ph - 2)
        c0 = rng.randint(1, w - pw - 2)
        box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        if all(_bboxes_far_enough(box, old, 3) for old in boxes):
            boxes.append(box)
            if len(boxes) == count:
                return boxes
    return None


def _marker_is_valid(
    coord: tuple[int, int],
    panels: list[Panel],
    markers: list[tuple[int, int, int]],
    shape: tuple[int, int],
) -> bool:
    r, c = coord
    if not in_bounds(shape, r, c):
        return False
    for panel in panels:
        if _inside_bbox(coord, panel.bbox):
            return False
        if sum(_contact_counts([coord], [panel]).values()) > 0:
            return False
    for mr, mc, _ in markers:
        if abs(mr - r) + abs(mc - c) <= 1:
            return False
    return True


def _copy_fits(copy_coords: list[tuple[int, int]], panels: list[Panel], shape: tuple[int, int]) -> bool:
    for coord in copy_coords:
        if not in_bounds(shape, coord[0], coord[1]):
            return False
        if any(_inside_bbox(coord, panel.bbox) for panel in panels):
            return False
    return True


def _find_anchor_for_constraints(
    rng: random.Random,
    panel: Panel,
    panels: list[Panel],
    shape: tuple[int, int],
    existing_markers: list[tuple[int, int, int]],
    *,
    desired_side: str | None = None,
    second_panel: Panel | None = None,
    second_side: str | None = None,
    require_no_contact: bool = False,
    allowed_side: str | None = None,
) -> tuple[int, int] | None:
    h, w = shape
    coords = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(coords)
    for anchor_coord in coords:
        if not _marker_is_valid(anchor_coord, panels, existing_markers, shape):
            continue
        copy1 = [(anchor_coord[0] + dr, anchor_coord[1] + dc) for dr, dc in panel.sprite_rel]
        if not _copy_fits(copy1, panels, shape):
            continue
        counts1 = _contact_counts(copy1, panels)
        if require_no_contact and max(counts1.values()) != 0:
            continue
        if desired_side is not None:
            if counts1[desired_side] == 0 or sum(counts1.values()) != counts1[desired_side]:
                continue
        if allowed_side is not None:
            if counts1[allowed_side] == 0 or sum(counts1.values()) != counts1[allowed_side]:
                continue
        if second_panel is not None:
            copy2 = [(anchor_coord[0] + dr, anchor_coord[1] + dc) for dr, dc in second_panel.sprite_rel]
            if not _copy_fits(copy2, panels, shape):
                continue
            counts2 = _contact_counts(copy2, panels)
            assert second_side is not None
            if counts2[second_side] == 0 or sum(counts2.values()) != counts2[second_side]:
                continue
        return anchor_coord
    return None


def _build_panel_objects(rng: random.Random, h: int, w: int, panel_count: int) -> tuple[np.ndarray, list[Panel]] | None:
    boxes = _sample_panel_bboxes(rng, h, w, panel_count)
    if boxes is None:
        return None

    anchor_colors = rng.sample(ANCHOR_COLORS, panel_count)
    panel_colors = rng.sample(PANEL_COLORS, panel_count)
    used = set(anchor_colors) | set(panel_colors)
    sprite_pool = [c for c in range(1, 10) if c not in used]
    sprite_colors = rng.sample(sprite_pool, panel_count)

    grid = np.zeros((h, w), dtype=int)
    panels = []
    for idx, box in enumerate(boxes):
        panel_color = panel_colors[idx]
        sprite_color = sprite_colors[idx]
        anchor_color = anchor_colors[idx]
        r0, c0, r1, c1 = box
        grid[r0 : r1 + 1, c0 : c1 + 1] = panel_color
        anchor, sprite_abs = _place_panel_contents(rng, box)
        for rr, cc in sprite_abs:
            grid[rr, cc] = sprite_color
        grid[anchor] = anchor_color
        sprite_rel = tuple(sorted((rr - anchor[0], cc - anchor[1]) for rr, cc in sprite_abs))
        panels.append(
            Panel(
                bbox=box,
                panel_color=panel_color,
                sprite_color=sprite_color,
                anchor_color=anchor_color,
                anchor=anchor,
                sprite_rel=sprite_rel,
            )
        )
    return grid, panels


def _intended_chain_length(rng: random.Random, panel_count: int) -> int:
    if panel_count == 2:
        return 2
    return 3 if rng.random() < 0.6 else 2


def _find_chain_plan(
    rng: random.Random,
    panels: list[Panel],
    markers: list[tuple[int, int, int]],
    shape: tuple[int, int],
    chain_len: int,
) -> tuple[list[int], tuple[int, int]] | None:
    orders = list(permutations(range(len(panels)), chain_len))
    rng.shuffle(orders)
    for order_tuple in orders:
        order = list(order_tuple)
        root = panels[order[0]]
        if chain_len == 3:
            mid = panels[order[1]]
            end = panels[order[2]]
            first_side = next(side for side, color in SIDE_TO_COLOR.items() if color == mid.anchor_color)
            second_side = next(side for side, color in SIDE_TO_COLOR.items() if color == end.anchor_color)
            root_anchor = _find_anchor_for_constraints(
                rng,
                root,
                panels,
                shape,
                markers,
                desired_side=first_side,
                second_panel=mid,
                second_side=second_side,
            )
        else:
            nxt = panels[order[1]]
            first_side = next(side for side, color in SIDE_TO_COLOR.items() if color == nxt.anchor_color)
            root_anchor = _find_anchor_for_constraints(
                rng,
                root,
                panels,
                shape,
                markers,
                desired_side=first_side,
            )
        if root_anchor is not None:
            return order, root_anchor
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(22, 28)
    w = rng.randint(22, 28)
    panel_count = 3 if rng.random() < 0.55 else 2
    built = _build_panel_objects(rng, h, w, panel_count)
    if built is None:
        return None
    grid, panels = built

    chain_len = _intended_chain_length(rng, panel_count)
    markers: list[tuple[int, int, int]] = []
    plan = _find_chain_plan(rng, panels, markers, grid.shape, chain_len)
    if plan is None and chain_len == 3:
        chain_len = 2
        plan = _find_chain_plan(rng, panels, markers, grid.shape, chain_len)
    if plan is None:
        return None
    order, root_anchor = plan
    root = panels[order[0]]
    markers.append((root_anchor[0], root_anchor[1], root.anchor_color))

    if rng.random() < 0.75:
        unused_sides = [side for side, color in SIDE_TO_COLOR.items() if color not in {p.anchor_color for p in panels}]
        if unused_sides and rng.random() < 0.6:
            extra_anchor = _find_anchor_for_constraints(
                rng,
                root,
                panels,
                grid.shape,
                markers,
                allowed_side=rng.choice(unused_sides),
            )
        else:
            extra_anchor = _find_anchor_for_constraints(
                rng,
                root,
                panels,
                grid.shape,
                markers,
                require_no_contact=True,
            )
        if extra_anchor is not None:
            markers.append((extra_anchor[0], extra_anchor[1], root.anchor_color))

    noise_colors = [c for c in range(1, 10) if c not in {p.anchor_color for p in panels}]
    noise_count = rng.randint(1, 3)
    for _ in range(noise_count):
        placed = False
        for _ in range(200):
            coord = (rng.randrange(h), rng.randrange(w))
            if _marker_is_valid(coord, panels, markers, grid.shape):
                markers.append((coord[0], coord[1], rng.choice(noise_colors)))
                placed = True
                break
        if not placed:
            return None

    for r, c, color in markers:
        grid[r, c] = color

    parsed_panels, parsed_markers = _parse_grid(grid)
    if len(parsed_panels) != panel_count or len(parsed_markers) != len(markers):
        return None

    out, meta = _simulate(grid)
    if len(meta["processed"]) != chain_len:
        return None
    if not any(copy["color"] in {p.anchor_color for p in parsed_panels} for copy in meta["copies"]):
        return None
    if not any(max(copy["counts"].values()) == 0 for copy in meta["copies"]):
        if not any(copy["color"] not in {p.anchor_color for p in parsed_panels} for copy in meta["copies"]):
            return None
    if np.array_equal(out, grid):
        return None

    panel_area = sum((p.bbox[2] - p.bbox[0] + 1) * (p.bbox[3] - p.bbox[1] + 1) for p in parsed_panels)
    if int(np.count_nonzero(out)) >= panel_area + 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        out = main(grid)
        assert isinstance(out, np.ndarray)
        assert grid.ndim == 2 and max(grid.shape) <= 30
        assert out.ndim == 2 and max(out.shape) <= 30
        return grid
    raise RuntimeError("failed to generate a valid cascade-stamp ARC input")
