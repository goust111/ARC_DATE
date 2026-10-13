from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


YELLOW = 4
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def label_components(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, list[np.ndarray]]:
    dirs = DIR4 if connectivity == 4 else DIR8
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comps: list[np.ndarray] = []
    next_id = 1

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            q = deque([(r, c)])
            labels[r, c] = next_id
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
            next_id += 1

    return labels, comps


def _line_offsets(orientation: str, anchor_end: str, length: int) -> list[tuple[int, int]]:
    sign = 1 if anchor_end in {"top", "left"} else -1
    if orientation == "v":
        return [(sign * t, 0) for t in range(1, length + 1)]
    return [(0, sign * t) for t in range(1, length + 1)]


def _mirror_offset(orientation: str, dr: int, dc: int) -> tuple[int, int]:
    if orientation == "v":
        return dr, -dc
    return -dr, dc


def _portal_frontier(portal: dict, side: str) -> list[tuple[int, int]]:
    if portal["orientation"] == "v":
        delta = -1 if side == "left" else 1
        return [(int(r), int(portal["col"]) + delta) for r in range(int(portal["r0"]), int(portal["r1"]) + 1)]
    delta = -1 if side == "up" else 1
    return [(int(portal["row"]) + delta, int(c)) for c in range(int(portal["c0"]), int(portal["c1"]) + 1)]


def _find_portals(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    portals: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg, YELLOW}):
        labels, comps = label_components(grid == color, connectivity=4)
        _ = labels
        for coords in comps:
            rows = coords[:, 0]
            cols = coords[:, 1]
            unique_rows = np.unique(rows)
            unique_cols = np.unique(cols)
            orientation = None
            portal = None

            if len(unique_cols) == 1:
                c = int(unique_cols[0])
                r0, r1 = int(rows.min()), int(rows.max())
                if len(coords) == r1 - r0 + 1 and np.array_equal(np.sort(rows), np.arange(r0, r1 + 1)):
                    markers = []
                    if r0 - 1 >= 0 and int(grid[r0 - 1, c]) == YELLOW:
                        markers.append(("top", (r0 - 1, c)))
                    if r1 + 1 < h and int(grid[r1 + 1, c]) == YELLOW:
                        markers.append(("bottom", (r1 + 1, c)))
                    if len(coords) >= 3 and len(markers) == 1:
                        orientation = "v"
                        anchor_end, yellow = markers[0]
                        portal = {
                            "color": int(color),
                            "orientation": orientation,
                            "r0": r0,
                            "r1": r1,
                            "col": c,
                            "length": int(len(coords)),
                            "anchor_end": anchor_end,
                            "yellow": tuple(map(int, yellow)),
                        }
            elif len(unique_rows) == 1:
                r = int(unique_rows[0])
                c0, c1 = int(cols.min()), int(cols.max())
                if len(coords) == c1 - c0 + 1 and np.array_equal(np.sort(cols), np.arange(c0, c1 + 1)):
                    markers = []
                    if c0 - 1 >= 0 and int(grid[r, c0 - 1]) == YELLOW:
                        markers.append(("left", (r, c0 - 1)))
                    if c1 + 1 < w and int(grid[r, c1 + 1]) == YELLOW:
                        markers.append(("right", (r, c1 + 1)))
                    if len(coords) >= 3 and len(markers) == 1:
                        orientation = "h"
                        anchor_end, yellow = markers[0]
                        portal = {
                            "color": int(color),
                            "orientation": orientation,
                            "row": r,
                            "c0": c0,
                            "c1": c1,
                            "length": int(len(coords)),
                            "anchor_end": anchor_end,
                            "yellow": tuple(map(int, yellow)),
                        }

            if portal is not None:
                portals.append(portal)

    assert portals, "expected at least one portal"
    return portals


def _detect_active_side(grid: np.ndarray, portal: dict, bg: int, portal_colors: set[int]) -> str:
    h, w = grid.shape
    if portal["orientation"] == "v":
        sides = ("left", "right")
    else:
        sides = ("up", "down")

    counts: dict[str, int] = {}
    for side in sides:
        count = 0
        for r, c in _portal_frontier(portal, side):
            if 0 <= r < h and 0 <= c < w:
                value = int(grid[r, c])
                if value != bg and value != YELLOW and value not in portal_colors:
                    count += 1
        counts[side] = count

    positives = [side for side, count in counts.items() if count > 0]
    assert len(positives) == 1, "portal must have exactly one active side"
    return positives[0]


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    portals = _find_portals(g, bg)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for portal in portals:
        by_color[int(portal["color"])].append(portal)

    assert all(len(items) == 2 for items in by_color.values()), "each portal color must appear exactly twice"
    portal_colors = set(by_color.keys())

    sprite_mask = (g != bg) & (g != YELLOW)
    for color in portal_colors:
        sprite_mask &= g != int(color)
    labels, comps = label_components(sprite_mask, connectivity=8)

    component_owner: dict[int, tuple[int, str]] = {}
    opposite = {"left": "right", "right": "left", "up": "down", "down": "up"}

    for portal in portals:
        active_side = _detect_active_side(g, portal, bg, portal_colors)
        portal["active_side"] = active_side
        touched_ids = set()
        for r, c in _portal_frontier(portal, active_side):
            if 0 <= r < g.shape[0] and 0 <= c < g.shape[1] and labels[r, c] != 0:
                touched_ids.add(int(labels[r, c]))
        assert touched_ids, "every portal must touch at least one sprite component"
        portal["component_ids"] = sorted(touched_ids)
        yr, yc = map(int, portal["yellow"])
        pixels = []
        for cid in portal["component_ids"]:
            if cid in component_owner:
                raise AssertionError("sprite component touches multiple portals")
            component_owner[cid] = (int(portal["color"]), str(active_side))
            for r, c in comps[cid - 1]:
                pixels.append((int(r) - yr, int(c) - yc, int(g[int(r), int(c)])))
        portal["pixels"] = pixels

    if comps:
        expected = set(range(1, len(comps) + 1))
        assert set(component_owner) == expected, "all non-portal objects must belong to a portal side"

    pairs = []
    for color, items in sorted(by_color.items()):
        a, b = items
        assert a["orientation"] == b["orientation"], "paired portals must share orientation"
        assert a["length"] == b["length"], "paired portals must share length"
        assert a["active_side"] == opposite[b["active_side"]], "paired portals must face opposite directions"
        pairs.append((a, b))

    return {
        "grid": g,
        "bg": int(bg),
        "portals": portals,
        "pairs": pairs,
        "sprite_components": comps,
    }


def _teleport(out: np.ndarray, src: dict, dst: dict) -> None:
    yr, yc = map(int, dst["yellow"])
    for dr, dc, color in src["pixels"]:
        nr, nc = _mirror_offset(dst["orientation"], int(dr), int(dc))
        out[yr + nr, yc + nc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the sprites beside each color-matched portal pair using yellow-end anchors and reflection across the portal line.

    Input structure:
    - bg is the most common color.
    - Several one-cell-thick horizontal or vertical portal segments appear on the background; each non-yellow portal color occurs in exactly two segments.
    - A single yellow pixel sits immediately beyond one end of every segment and marks that portal's anchor end.
    - On exactly one side of each segment, one or more multicolor sprites touch the cells next to the portal; the opposite side stays empty.

    Rule:
    - Pair the two portals of each line color.
    - For every sprite pixel beside portal A, measure its row and column offset from A's yellow anchor.
    - Recreate those pixels at portal B with the same anchor-relative offset, but reflect the offset across B's portal line so only the perpendicular coordinate changes sign.
    - Do the same from B to A, clear the original sprite pixels to bg, and leave portal lines plus yellow anchors unchanged.

    Key insight:
    - The transformation depends on combining three visible cues at once: same-color line pairing, yellow-end anchor alignment, and reflection across the portal axis because the two active sides face opposite directions.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    out = np.array(scene["grid"], dtype=int, copy=True)

    for coords in scene["sprite_components"]:
        out[coords[:, 0], coords[:, 1]] = int(scene["bg"])

    for a, b in scene["pairs"]:
        _teleport(out, a, b)
        _teleport(out, b, a)

    return out


SPRITE_TEMPLATES = [
    [(0, 1), (1, 1)],
    [(0, 1), (0, 2)],
    [(0, 1), (1, 1), (1, 2)],
    [(0, 1), (1, 1), (0, 2)],
    [(0, 1), (1, 1), (0, 2), (1, 2)],
    [(0, 1), (1, 1), (2, 1)],
    [(0, 1), (1, 1), (2, 1), (1, 2)],
    [(0, 1), (0, 2), (0, 3)],
]
MAX_RETRIES = 300


def _transform_side_cell(
    orientation: str,
    anchor_end: str,
    active_side: str,
    t: int,
    d: int,
) -> tuple[int, int]:
    line_sign = 1 if anchor_end in {"top", "left"} else -1
    side_sign = -1 if active_side in {"left", "up"} else 1
    if orientation == "v":
        return line_sign * t, side_sign * d
    return side_sign * d, line_sign * t


def _generate_side_pixels(
    length: int,
    orientation: str,
    anchor_end: str,
    active_side: str,
    sprite_palette: list[int],
) -> list[tuple[int, int, int]] | None:
    max_components = min(3, max(1, (length + 1) // 2))
    n_components = random.randint(1, max_components)
    abstract_cells: list[set[tuple[int, int]]] = []
    occupied: set[tuple[int, int]] = set()

    for _ in range(n_components):
        placed = False
        for _ in range(80):
            template = random.choice(SPRITE_TEMPLATES)
            max_t = max(t for t, _ in template)
            if max_t >= length:
                continue
            start_t = random.randint(1, length - max_t)
            cells = {(start_t + dt, dd) for dt, dd in template}
            if any(max(abs(t1 - t2), abs(d1 - d2)) <= 1 for t1, d1 in cells for t2, d2 in occupied):
                continue
            abstract_cells.append(cells)
            occupied |= cells
            placed = True
            break
        if not placed:
            return None

    pixels: list[tuple[int, int, int]] = []
    for cells in abstract_cells:
        primary = random.choice(sprite_palette)
        area = len(cells)
        use_secondary = area >= 3 and random.random() < 0.5
        secondary = primary
        if use_secondary:
            secondary_choices = [color for color in sprite_palette if color != primary]
            secondary = random.choice(secondary_choices) if secondary_choices else primary
            accent_count = random.randint(1, min(2, area - 1))
            accent_cells = set(random.sample(list(cells), accent_count))
        else:
            accent_cells = set()

        for t, d in sorted(cells):
            color = secondary if (t, d) in accent_cells else primary
            dr, dc = _transform_side_cell(orientation, anchor_end, active_side, int(t), int(d))
            pixels.append((int(dr), int(dc), int(color)))

    return pixels


def _reserve_offsets(portal: dict, partner_pixels: list[tuple[int, int, int]]) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    cells.update(_line_offsets(portal["orientation"], portal["anchor_end"], portal["length"]))
    cells.update((int(dr), int(dc)) for dr, dc, _ in portal["pixels"])
    for dr, dc, _ in partner_pixels:
        cells.add(_mirror_offset(portal["orientation"], int(dr), int(dc)))
    return cells


def _pad_offsets(offsets: set[tuple[int, int]]) -> set[tuple[int, int]]:
    padded = set()
    for r, c in offsets:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                padded.add((int(r) + dr, int(c) + dc))
    return padded


def _bbox(offsets: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in offsets]
    cols = [c for _, c in offsets]
    return min(rows), min(cols), max(rows), max(cols)


def _build_pair(portal_color: int, sprite_palette: list[int]) -> tuple[dict, dict] | None:
    orientation = random.choice(["v", "h"])
    length = random.randint(4, 8)

    if orientation == "v":
        side_a = random.choice(["left", "right"])
        side_b = "right" if side_a == "left" else "left"
        anchor_names = ["top", "bottom"]
    else:
        side_a = random.choice(["up", "down"])
        side_b = "down" if side_a == "up" else "up"
        anchor_names = ["left", "right"]

    anchor_a = random.choice(anchor_names)
    anchor_b = random.choice(anchor_names)
    pixels_a = _generate_side_pixels(length, orientation, anchor_a, side_a, sprite_palette)
    pixels_b = _generate_side_pixels(length, orientation, anchor_b, side_b, sprite_palette)
    if pixels_a is None or pixels_b is None:
        return None

    portal_a = {
        "color": int(portal_color),
        "orientation": orientation,
        "length": int(length),
        "anchor_end": anchor_a,
        "active_side": side_a,
        "pixels": pixels_a,
    }
    portal_b = {
        "color": int(portal_color),
        "orientation": orientation,
        "length": int(length),
        "anchor_end": anchor_b,
        "active_side": side_b,
        "pixels": pixels_b,
    }

    reserve_a = _reserve_offsets(portal_a, pixels_b)
    reserve_b = _reserve_offsets(portal_b, pixels_a)
    portal_a["line_offsets"] = _line_offsets(orientation, anchor_a, length)
    portal_b["line_offsets"] = _line_offsets(orientation, anchor_b, length)
    portal_a["reserve_offsets"] = reserve_a
    portal_b["reserve_offsets"] = reserve_b
    portal_a["padded_offsets"] = _pad_offsets(reserve_a)
    portal_b["padded_offsets"] = _pad_offsets(reserve_b)
    portal_a["padded_bbox"] = _bbox(portal_a["padded_offsets"])
    portal_b["padded_bbox"] = _bbox(portal_b["padded_offsets"])
    return portal_a, portal_b


def _place_portals(portals: list[dict], h: int, w: int) -> bool:
    blocked: set[tuple[int, int]] = set()

    for portal in sorted(portals, key=lambda p: len(p["padded_offsets"]), reverse=True):
        r0, c0, r1, c1 = portal["padded_bbox"]
        min_y = -r0
        max_y = h - 1 - r1
        min_x = -c0
        max_x = w - 1 - c1
        if min_y > max_y or min_x > max_x:
            return False

        placed = False
        for _ in range(200):
            yr = random.randint(min_y, max_y)
            yc = random.randint(min_x, max_x)
            abs_padded = {(yr + dr, yc + dc) for dr, dc in portal["padded_offsets"]}
            if abs_padded & blocked:
                continue
            portal["yellow"] = (int(yr), int(yc))
            portal["placed_line"] = [(int(yr + dr), int(yc + dc)) for dr, dc in portal["line_offsets"]]
            portal["placed_pixels"] = [(int(yr + dr), int(yc + dc), int(color)) for dr, dc, color in portal["pixels"]]
            portal["blocked_abs"] = abs_padded
            blocked |= abs_padded
            placed = True
            break
        if not placed:
            return False

    return True


def _fill_portal_metadata(portal: dict) -> None:
    yr, yc = map(int, portal["yellow"])
    if portal["orientation"] == "v":
        rows = [r for r, _ in portal["placed_line"]]
        portal["r0"] = int(min(rows))
        portal["r1"] = int(max(rows))
        portal["col"] = int(portal["placed_line"][0][1])
    else:
        cols = [c for _, c in portal["placed_line"]]
        portal["row"] = int(portal["placed_line"][0][0])
        portal["c0"] = int(min(cols))
        portal["c1"] = int(max(cols))
    _ = (yr, yc)


def _draw_scene(grid: np.ndarray, portals: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for portal in portals:
        yr, yc = map(int, portal["yellow"])
        out[yr, yc] = YELLOW
        for r, c in portal["placed_line"]:
            out[int(r), int(c)] = int(portal["color"])
        for r, c, color in portal["placed_pixels"]:
            out[int(r), int(c)] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    bg_choices = [color for color in range(10) if color != YELLOW]
    bg = random.choice(bg_choices)
    available = [color for color in range(10) if color not in {bg, YELLOW}]

    num_pairs = random.randint(1, 3)
    random.shuffle(available)
    portal_colors = available[:num_pairs]
    sprite_palette = available[num_pairs:]
    if len(sprite_palette) < 2:
        return None

    pairs = []
    for portal_color in portal_colors:
        built = None
        for _ in range(40):
            built = _build_pair(int(portal_color), sprite_palette)
            if built is not None:
                break
        if built is None:
            return None
        pairs.append(built)

    portals = [portal for pair in pairs for portal in pair]
    max_box_h = max(portal["padded_bbox"][2] - portal["padded_bbox"][0] + 1 for portal in portals)
    max_box_w = max(portal["padded_bbox"][3] - portal["padded_bbox"][1] + 1 for portal in portals)
    base_h = max(int(max_box_h + 2), 14 + 3 * (num_pairs - 1))
    base_w = max(int(max_box_w + 2), 14 + 3 * (num_pairs - 1))
    if base_h > 30 or base_w > 30:
        return None

    h = random.randint(base_h, 30)
    w = random.randint(base_w, 30)
    if not _place_portals(portals, h, w):
        return None

    for portal in portals:
        _fill_portal_metadata(portal)

    grid = np.full((h, w), int(bg), dtype=int)
    grid = _draw_scene(grid, portals)

    try:
        scene = _parse_scene(grid)
        out = main(grid)
    except AssertionError:
        return None

    if len(scene["pairs"]) != num_pairs:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid portal transport input")
