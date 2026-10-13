from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
BLUE = 1
GREEN = 3
CONNECTOR_COLORS = (2, 4, 7)

MAX_GEN_ATTEMPTS = 500
SHAPE_ATTEMPTS = 250


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _tight_crop(grid: np.ndarray, bg: int = BG) -> np.ndarray:
    coords = np.argwhere(grid != bg)
    if coords.size == 0:
        return np.full((1, 1), bg, dtype=int)
    return _crop_bbox(grid, _bbox_coords(coords))


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _connected_components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _extract_objects(grid: np.ndarray, body_color: int, side: str) -> list[dict]:
    h, w = grid.shape
    objects = []
    for coords in _connected_components_of_color(grid, body_color):
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        if side == "right":
            connector_col = c1 + 1
            if connector_col >= w:
                raise AssertionError("blue connector exits grid")
            edge_rows = {int(r) for r, c in coords if int(c) == c1}
            candidate_rows = [r for r in edge_rows if int(grid[r, connector_col]) in CONNECTOR_COLORS]
            if len(candidate_rows) != 1:
                raise AssertionError("blue body must have exactly one right connector")
            connector = (int(candidate_rows[0]), int(connector_col))
        else:
            connector_col = c0 - 1
            if connector_col < 0:
                raise AssertionError("green connector exits grid")
            edge_rows = {int(r) for r, c in coords if int(c) == c0}
            candidate_rows = [r for r in edge_rows if int(grid[r, connector_col]) in CONNECTOR_COLORS]
            if len(candidate_rows) != 1:
                raise AssertionError("green body must have exactly one left connector")
            connector = (int(candidate_rows[0]), int(connector_col))

        objects.append(
            {
                "coords": coords,
                "bbox": bbox,
                "connector": connector,
                "connector_color": int(grid[connector]),
            }
        )

    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    return objects


def _analyze_scene(grid: np.ndarray) -> tuple[list[dict], list[dict], list[tuple[int, int]]]:
    blues = _extract_objects(grid, BLUE, "right")
    greens = _extract_objects(grid, GREEN, "left")

    for blue in blues:
        shift = _bbox_w(blue["bbox"])
        blue["shift"] = int(shift)
        blue["shifted_coords"] = blue["coords"] + np.array([0, shift], dtype=int)
        br, bc = blue["connector"]
        blue["shifted_connector"] = (int(br), int(bc + shift))

    matches: list[tuple[int, int]] = []
    blue_match_count = Counter()
    green_match_count = Counter()
    for bi, blue in enumerate(blues):
        br, bc = blue["shifted_connector"]
        for gi, green in enumerate(greens):
            gr, gc = green["connector"]
            if int(blue["connector_color"]) != int(green["connector_color"]):
                continue
            if int(br) != int(gr):
                continue
            if abs(int(bc) - int(gc)) != 1:
                continue
            matches.append((bi, gi))
            blue_match_count[bi] += 1
            green_match_count[gi] += 1

    if any(count > 1 for count in blue_match_count.values()):
        raise AssertionError("a blue sprite matches multiple green sprites")
    if any(count > 1 for count in green_match_count.values()):
        raise AssertionError("a green sprite matches multiple blue sprites")

    return blues, greens, matches


def main(grid: np.ndarray) -> np.ndarray:
    """Shift blue sprites right by their own width, keep only connector-matched docked blue-green pairs, then crop.

    Input structure:
    - The grid has black background and several 4-connected monochrome blue (1) and green (3) sprite bodies.
    - Every blue body has exactly one colored connector pixel immediately outside its right bbox edge.
    - Every green body has exactly one colored connector pixel immediately outside its left bbox edge.

    Rule:
    - Treat each blue body together with its connector as one movable sprite and shift it right by its own bbox width.
    - Green sprites stay fixed.
    - After the shift, a blue-green pair survives only when their connector pixels are on the same row, horizontally adjacent, and the two connector colors are equal.
    - For every surviving pair, recolor the shifted blue connector to blue and the green connector to green, erase every unmatched sprite, and tightly crop the remaining non-background pixels.

    Key insight:
    - The decisive offset is local: each blue body moves by its own width, so only some shifted connector pairs become color-matched docking neighbors.
    """
    g = np.asarray(grid, dtype=int)
    blues, greens, matches = _analyze_scene(g)

    if not matches:
        return np.full((1, 1), BG, dtype=int)

    out = np.full_like(g, BG)
    for blue_idx, green_idx in matches:
        blue = blues[blue_idx]
        green = greens[green_idx]

        shifted = blue["shifted_coords"]
        out[shifted[:, 0], shifted[:, 1]] = BLUE
        br, bc = blue["shifted_connector"]
        out[int(br), int(bc)] = BLUE

        coords = green["coords"]
        out[coords[:, 0], coords[:, 1]] = GREEN
        gr, gc = green["connector"]
        out[int(gr), int(gc)] = GREEN

    return _tight_crop(out, BG)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    return _bbox_coords(coords)


def _is_connected_mask(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(int(v) for v in coords[0])
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    queue = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                queue.append((nr, nc))

    return bool((~mask & ~outside).any())


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = set()
    for r, c in np.argwhere(mask):
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                cells.add((nr, nc))
    return list(cells)


def _draw_random_manhattan_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    tr, tc = end
    mask[int(r), int(c)] = True
    while (r, c) != (tr, tc):
        moves = []
        if r < tr:
            moves.append((1, 0))
        elif r > tr:
            moves.append((-1, 0))
        if c < tc:
            moves.append((0, 1))
        elif c > tc:
            moves.append((0, -1))
        dr, dc = random.choice(moves)
        r += dr
        c += dc
        mask[int(r), int(c)] = True


def _tight_crop_mask(mask: np.ndarray) -> np.ndarray:
    return _crop_bbox(mask.astype(int), _mask_bbox(mask)).astype(bool)


def _random_body_with_width(width: int) -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        canvas_h = random.randint(2, 4)
        mask = np.zeros((canvas_h, width), dtype=bool)

        left_seed = (random.randrange(canvas_h), 0)
        right_seed = (random.randrange(canvas_h), width - 1)
        _draw_random_manhattan_path(mask, left_seed, right_seed)

        while _mask_bbox(mask)[2] == _mask_bbox(mask)[0] and int(mask.sum()) < min(10, canvas_h * width):
            choices = [cell for cell in _frontier(mask) if cell[0] != _mask_bbox(mask)[0]]
            if not choices:
                break
            rr, cc = random.choice(choices)
            mask[rr, cc] = True

        min_area = max(4, int(mask.sum()))
        max_area = min(10, canvas_h * width)
        if min_area > max_area:
            continue
        target_area = random.randint(min_area, max_area)

        while int(mask.sum()) < target_area:
            frontier = _frontier(mask)
            if not frontier:
                break
            weighted = []
            for cell in frontier:
                nr, nc = cell
                score = 1 + sum(
                    1
                    for ar, ac in _neighbors4(nr, nc)
                    if 0 <= ar < mask.shape[0] and 0 <= ac < mask.shape[1] and mask[ar, ac]
                )
                weighted.extend([cell] * score)
            rr, cc = random.choice(weighted)
            mask[rr, cc] = True

        cropped = _tight_crop_mask(mask)
        if cropped.shape[1] != width:
            continue
        if cropped.shape[0] < 2:
            continue
        if not (4 <= int(cropped.sum()) <= 10):
            continue
        if not _is_connected_mask(cropped):
            continue
        if _has_hole(cropped):
            continue
        return cropped

    raise RuntimeError("failed to generate a valid sprite body")


def _mask_coords(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    return np.argwhere(mask) + np.array([top, left], dtype=int)


def _build_lane(kind: str, blue_width: int) -> dict:
    blue_mask = _random_body_with_width(blue_width)
    green_mask = _random_body_with_width(random.randint(2, 5))

    blue_edge_rows = np.where(blue_mask[:, -1])[0].tolist()
    green_edge_rows = np.where(green_mask[:, 0])[0].tolist()
    blue_conn_r = int(random.choice(blue_edge_rows))
    green_conn_r = int(random.choice(green_edge_rows))

    align_row = max(blue_conn_r, green_conn_r)
    blue_top = int(align_row - blue_conn_r)
    green_top = int(align_row - green_conn_r)

    shift = int(blue_mask.shape[1])
    connector_gap = 1 if kind in {"match", "mismatch_color"} else random.choice([2, 3])
    green_left = int(2 * shift + connector_gap + 1)

    if kind == "mismatch_color":
        blue_connector_color = random.choice(CONNECTOR_COLORS)
        green_connector_color = random.choice([c for c in CONNECTOR_COLORS if c != blue_connector_color])
    else:
        blue_connector_color = green_connector_color = random.choice(CONNECTOR_COLORS)

    return {
        "kind": kind,
        "height": max(blue_top + blue_mask.shape[0], green_top + green_mask.shape[0]),
        "width": green_left + green_mask.shape[1],
        "blue": {
            "mask": blue_mask,
            "top": blue_top,
            "left": 0,
            "connector": (align_row, shift),
            "connector_color": blue_connector_color,
            "shift": shift,
        },
        "green": {
            "mask": green_mask,
            "top": green_top,
            "left": green_left,
            "connector": (align_row, green_left - 1),
            "connector_color": green_connector_color,
        },
    }


def _place_lanes(lanes: list[dict]) -> tuple[np.ndarray, list[dict]]:
    ordered = lanes[:]
    random.shuffle(ordered)

    gaps = [random.randint(1, 3) for _ in range(max(0, len(ordered) - 1))]
    packed_h = sum(lane["height"] for lane in ordered) + sum(gaps)
    packed_w = max(lane["width"] for lane in ordered)

    h_low = max(16, packed_h)
    h_high = min(30, packed_h + 5)
    w_low = max(18, packed_w)
    w_high = min(30, packed_w + 6)
    if h_low > h_high or w_low > w_high:
        raise RuntimeError("scene does not fit ARC bounds")

    height = random.randint(h_low, h_high)
    width = random.randint(w_low, w_high)
    top_margin = random.randint(0, height - packed_h)

    grid = np.full((height, width), BG, dtype=int)
    placed: list[dict] = []

    row = top_margin
    for idx, lane in enumerate(ordered):
        col = random.randint(0, width - lane["width"])
        lane_record = {"kind": lane["kind"]}

        for body_color, key in ((BLUE, "blue"), (GREEN, "green")):
            body = lane[key]
            coords = _mask_coords(body["mask"], row + body["top"], col + body["left"])
            grid[coords[:, 0], coords[:, 1]] = body_color
            cr, cc = body["connector"]
            connector = (int(row + cr), int(col + cc))
            grid[connector] = int(body["connector_color"])
            lane_record[key] = {
                "coords": coords,
                "bbox": _bbox_coords(coords),
                "connector": connector,
                "connector_color": int(body["connector_color"]),
                "shift": int(body.get("shift", 0)),
            }

        placed.append(lane_record)
        if idx < len(gaps):
            row += lane["height"] + gaps[idx]

    return grid, placed


def _render_expected(shape: tuple[int, int], placed_lanes: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for lane in placed_lanes:
        if lane["kind"] != "match":
            continue
        blue = lane["blue"]
        green = lane["green"]

        shifted = blue["coords"] + np.array([0, blue["shift"]], dtype=int)
        out[shifted[:, 0], shifted[:, 1]] = BLUE
        br, bc = blue["connector"]
        out[int(br), int(bc + blue["shift"])] = BLUE

        out[green["coords"][:, 0], green["coords"][:, 1]] = GREEN
        gr, gc = green["connector"]
        out[int(gr), int(gc)] = GREEN

    return _tight_crop(out, BG)


def _validate_case(inp: np.ndarray, expected: np.ndarray, placed_lanes: list[dict]) -> bool:
    if len(placed_lanes) not in (2, 3):
        return False
    if sum(1 for lane in placed_lanes if lane["kind"] == "match") not in (1, 2):
        return False
    if not any(lane["kind"] != "match" for lane in placed_lanes):
        return False

    blue_widths = [int(lane["blue"]["shift"]) for lane in placed_lanes]
    if len(set(blue_widths)) != len(blue_widths):
        return False

    try:
        blues, greens, matches = _analyze_scene(inp)
    except AssertionError:
        return False

    if len(blues) != len(placed_lanes) or len(greens) != len(placed_lanes):
        return False
    if len(matches) != sum(1 for lane in placed_lanes if lane["kind"] == "match"):
        return False

    parsed_blues = {tuple(obj["connector"]): obj for obj in blues}
    parsed_greens = {tuple(obj["connector"]): obj for obj in greens}
    match_keys = {
        (tuple(blues[blue_idx]["connector"]), tuple(greens[green_idx]["connector"]))
        for blue_idx, green_idx in matches
    }

    decoy_kinds = {lane["kind"] for lane in placed_lanes if lane["kind"] != "match"}
    if not decoy_kinds <= {"mismatch_color", "wrong_gap"}:
        return False

    for lane in placed_lanes:
        blue_key = tuple(lane["blue"]["connector"])
        green_key = tuple(lane["green"]["connector"])
        if blue_key not in parsed_blues or green_key not in parsed_greens:
            return False

        blue = parsed_blues[blue_key]
        green = parsed_greens[green_key]
        br, bc = blue["shifted_connector"]
        gr, gc = green["connector"]
        same_row = int(br) == int(gr)
        gap = abs(int(bc) - int(gc))
        same_color = int(blue["connector_color"]) == int(green["connector_color"])
        matched = (blue_key, green_key) in match_keys

        if lane["kind"] == "match":
            if not (same_row and gap == 1 and same_color and matched):
                return False
        elif lane["kind"] == "mismatch_color":
            if not (same_row and gap == 1 and not same_color and not matched):
                return False
        elif lane["kind"] == "wrong_gap":
            if not (same_row and gap > 1 and same_color and not matched):
                return False
        else:
            return False

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False
    if expected.ndim != 2 or max(expected.shape) > 30:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choices([1, 2], weights=[1, 3], k=1)[0]
    decoy_kind = random.choice(["mismatch_color", "wrong_gap"])
    lane_kinds = ["match"] * pair_count + [decoy_kind]
    blue_widths = random.sample([2, 3, 4, 5], len(lane_kinds))
    lanes = [_build_lane(kind, width) for kind, width in zip(lane_kinds, blue_widths)]

    try:
        inp, placed_lanes = _place_lanes(lanes)
    except RuntimeError:
        return None

    expected = _render_expected(inp.shape, placed_lanes)
    if not _validate_case(inp, expected, placed_lanes):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
