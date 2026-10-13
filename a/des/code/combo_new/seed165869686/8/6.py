import random
from collections import Counter, defaultdict

import numpy as np


BLACK = 0
GEN_RETRIES = 400
ROOM_CONFIGS = (
    (1, 2, "horizontal"),
    (1, 4, "horizontal"),
    (2, 2, "horizontal"),
    (2, 3, "horizontal"),
    (2, 4, "horizontal"),
    (2, 2, "vertical"),
    (3, 2, "vertical"),
    (4, 1, "vertical"),
)
PAIR_STATES = ("none", "first", "second", "both")


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _classify_wall_segment(segment: np.ndarray, bg: int) -> tuple[str, int | None]:
    vals = [int(v) for v in segment.tolist()]
    if all(v == BLACK for v in vals):
        return "wall", None
    if len(vals) < 4:
        return "invalid", None

    hits: list[int] = []
    colors: list[int] = []
    for start in range(len(vals) - 3):
        color = int(vals[start])
        if color in (BLACK, bg):
            continue
        if vals[start + 1] != bg or vals[start + 2] != bg or vals[start + 3] != color:
            continue
        if any(v != BLACK for v in vals[:start]) or any(v != BLACK for v in vals[start + 4 :]):
            continue
        hits.append(start)
        colors.append(color)

    if len(hits) == 1:
        return "portal", int(colors[0])
    return "invalid", None


def _try_parse_structure(g: np.ndarray, bg: int, ih: int, iw: int) -> dict | None:
    H, W = g.shape
    if (H - 1) % (ih + 1) != 0 or (W - 1) % (iw + 1) != 0:
        return None

    room_rows = (H - 1) // (ih + 1)
    room_cols = (W - 1) // (iw + 1)
    if room_rows < 1 or room_cols < 1:
        return None

    wall_rows = [k * (ih + 1) for k in range(room_rows + 1)]
    wall_cols = [k * (iw + 1) for k in range(room_cols + 1)]
    for wr in wall_rows:
        for wc in wall_cols:
            if int(g[wr, wc]) != BLACK:
                return None

    rooms = []
    portal_to_rooms: defaultdict[int, list[int]] = defaultdict(list)
    for rr in range(room_rows):
        for cc in range(room_cols):
            r0 = rr * (ih + 1)
            c0 = cc * (iw + 1)
            r1 = r0 + ih + 1
            c1 = c0 + iw + 1

            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            if interior.shape != (ih, iw) or np.any(interior == BLACK):
                return None

            sides = {
                "top": _classify_wall_segment(g[r0, c0 + 1 : c1], bg),
                "bottom": _classify_wall_segment(g[r1, c0 + 1 : c1], bg),
                "left": _classify_wall_segment(g[r0 + 1 : r1, c0], bg),
                "right": _classify_wall_segment(g[r0 + 1 : r1, c1], bg),
            }
            if any(kind == "invalid" for kind, _ in sides.values()):
                return None

            portal_sides = [(side, color) for side, (kind, color) in sides.items() if kind == "portal"]
            if len(portal_sides) != 1:
                return None
            portal_side, portal_color = portal_sides[0]

            payloads: list[tuple[int, int, int]] = []
            for pr, pc in np.argwhere(interior != bg):
                color = int(interior[pr, pc])
                if color == portal_color:
                    return None
                payloads.append((int(pr), int(pc), color))

            room_idx = len(rooms)
            rooms.append(
                {
                    "idx": room_idx,
                    "grid_rc": (rr, cc),
                    "r0": int(r0),
                    "c0": int(c0),
                    "portal_side": portal_side,
                    "portal_color": int(portal_color),
                    "payloads": payloads,
                }
            )
            portal_to_rooms[int(portal_color)].append(room_idx)

    if not rooms or any(len(v) != 2 for v in portal_to_rooms.values()):
        return None
    if len(portal_to_rooms) * 2 != len(rooms):
        return None

    return {
        "bg": int(bg),
        "ih": int(ih),
        "iw": int(iw),
        "room_rows": int(room_rows),
        "room_cols": int(room_cols),
        "rooms": rooms,
        "portal_to_rooms": {int(k): tuple(v) for k, v in portal_to_rooms.items()},
    }


def _extract_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    H, W = g.shape

    candidates = []
    for ih in range(1, H - 1):
        if (H - 1) % (ih + 1) != 0:
            continue
        for iw in range(1, W - 1):
            candidate = _try_parse_structure(g, bg, ih, iw)
            if candidate is not None:
                candidates.append(candidate)

    assert candidates, "failed to infer a valid room tiling"
    candidates.sort(
        key=lambda d: (
            -(d["room_rows"] * d["room_cols"]),
            -(d["ih"] * d["iw"]),
            -d["ih"],
            -d["iw"],
        )
    )
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Teleport each room's interior payload into its paired portal room after a 180-degree turn.

    Input structure:
    - The background color is the most common color.
    - Black 1-pixel walls tile the whole grid into equal rectangular rooms.
    - Each room has exactly one wall segment containing a portal pattern:
      one non-black, non-background color, then a 2-cell gap, then the same color.
    - Portal colors occur on exactly two rooms, defining one room-to-room pairing.
    - Some room interiors contain scattered non-background payload pixels that are
      different from that room's portal color.

    Rule:
    - Recover the room lattice and read each room's unique portal color from its
      wall-side portal pattern.
    - Pair rooms that share the same portal color.
    - Remove every interior payload pixel from its source room.
    - For each source-room payload cell at relative coordinate (r, c), place the
      same color into the paired room at (H-1-r, W-1-c), where H and W are the
      room-interior height and width.
    - All transfers happen simultaneously; black walls and portal markings stay unchanged.

    Key insight:
    - The colored wall cells are not ordinary objects: they label portals, and the
      payload does not simply translate between paired rooms but reappears after a
      180-degree point reflection inside the destination room.
    """
    g = np.asarray(grid, dtype=int)
    info = _extract_structure(g)
    bg = int(info["bg"])
    ih = int(info["ih"])
    iw = int(info["iw"])
    rooms = info["rooms"]

    out = g.copy()
    partner_of: dict[int, int] = {}
    for room_a, room_b in info["portal_to_rooms"].values():
        partner_of[int(room_a)] = int(room_b)
        partner_of[int(room_b)] = int(room_a)

    for room in rooms:
        base_r = int(room["r0"]) + 1
        base_c = int(room["c0"]) + 1
        for pr, pc, _color in room["payloads"]:
            out[base_r + int(pr), base_c + int(pc)] = bg

    for room in rooms:
        src_idx = int(room["idx"])
        dst = rooms[int(partner_of[src_idx])]
        dst_r = int(dst["r0"]) + 1
        dst_c = int(dst["c0"]) + 1
        for pr, pc, color in room["payloads"]:
            rr = ih - 1 - int(pr)
            cc = iw - 1 - int(pc)
            out[dst_r + rr, dst_c + cc] = int(color)

    return out


def _feasible_configs() -> list[tuple[int, int, str, int, int]]:
    configs = []
    for room_rows, room_cols, axis in ROOM_CONFIGS:
        ih_max = min(7, (29 // room_rows) - 1)
        iw_max = min(7, (29 // room_cols) - 1)
        if ih_max >= 4 and iw_max >= 4:
            configs.append((room_rows, room_cols, axis, ih_max, iw_max))
    return configs


GEN_CONFIGS = _feasible_configs()


def _portal_side(room_rows: int, room_cols: int, axis: str, rr: int, cc: int, rng: random.Random) -> str:
    if axis == "horizontal":
        if room_rows == 1:
            return rng.choice(["top", "bottom"])
        return "top" if rr == 0 else "bottom"
    if room_cols == 1:
        return rng.choice(["left", "right"])
    return "left" if cc == 0 else "right"


def _carve_portal(
    grid: np.ndarray,
    bg: int,
    r0: int,
    c0: int,
    ih: int,
    iw: int,
    side: str,
    color: int,
    rng: random.Random,
) -> None:
    if side == "top":
        start = c0 + 1 + rng.randint(0, iw - 4)
        grid[r0, start] = color
        grid[r0, start + 1 : start + 3] = bg
        grid[r0, start + 3] = color
    elif side == "bottom":
        row = r0 + ih + 1
        start = c0 + 1 + rng.randint(0, iw - 4)
        grid[row, start] = color
        grid[row, start + 1 : start + 3] = bg
        grid[row, start + 3] = color
    elif side == "left":
        start = r0 + 1 + rng.randint(0, ih - 4)
        grid[start, c0] = color
        grid[start + 1 : start + 3, c0] = bg
        grid[start + 3, c0] = color
    else:
        col = c0 + iw + 1
        start = r0 + 1 + rng.randint(0, ih - 4)
        grid[start, col] = color
        grid[start + 1 : start + 3, col] = bg
        grid[start + 3, col] = color


def _paint_payload(
    grid: np.ndarray,
    room: dict,
    ih: int,
    iw: int,
    colors: list[int],
    rng: random.Random,
) -> None:
    max_pixels = min(5, max(1, (ih * iw) // 4))
    count = rng.randint(1, max_pixels)
    positions = rng.sample([(r, c) for r in range(ih) for c in range(iw)], count)
    use_colors = rng.sample(colors, rng.randint(1, min(3, len(colors))))

    payloads = []
    for pr, pc in positions:
        color = rng.choice(use_colors)
        grid[int(room["r0"]) + 1 + pr, int(room["c0"]) + 1 + pc] = color
        payloads.append((int(pr), int(pc), int(color)))
    room["payloads"] = payloads


def _expected_from_metadata(grid: np.ndarray, bg: int, ih: int, iw: int, rooms: list[dict], pairs: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    partner_of: dict[int, int] = {}
    for a, b in pairs:
        partner_of[int(a)] = int(b)
        partner_of[int(b)] = int(a)

    for room in rooms:
        base_r = int(room["r0"]) + 1
        base_c = int(room["c0"]) + 1
        for pr, pc, _color in room["payloads"]:
            out[base_r + int(pr), base_c + int(pc)] = bg

    for room in rooms:
        dst = rooms[int(partner_of[int(room["idx"])])]
        dst_r = int(dst["r0"]) + 1
        dst_c = int(dst["c0"]) + 1
        for pr, pc, color in room["payloads"]:
            out[dst_r + ih - 1 - int(pr), dst_c + iw - 1 - int(pc)] = int(color)

    return out


def _bg_is_unique_mode(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.flatten()))
    top = max(counts.values())
    return counts[int(bg)] == top and list(counts.values()).count(top) == 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_rows, room_cols, axis, ih_max, iw_max = rng.choice(GEN_CONFIGS)
    ih = rng.randint(4, ih_max)
    iw = rng.randint(4, iw_max)
    H = 1 + room_rows * (ih + 1)
    W = 1 + room_cols * (iw + 1)

    bg = rng.choice([c for c in range(1, 10)])
    grid = np.full((H, W), bg, dtype=int)

    for r in range(0, H, ih + 1):
        grid[r, :] = BLACK
    for c in range(0, W, iw + 1):
        grid[:, c] = BLACK

    rooms = []
    for rr in range(room_rows):
        for cc in range(room_cols):
            rooms.append(
                {
                    "idx": len(rooms),
                    "r0": rr * (ih + 1),
                    "c0": cc * (iw + 1),
                    "grid_rc": (rr, cc),
                    "portal_side": _portal_side(room_rows, room_cols, axis, rr, cc, rng),
                    "payloads": [],
                }
            )

    room_indices = list(range(len(rooms)))
    rng.shuffle(room_indices)
    pairs = [(room_indices[i], room_indices[i + 1]) for i in range(0, len(room_indices), 2)]
    portal_colors = rng.sample([c for c in range(1, 10) if c != bg], len(pairs))

    pair_states = [rng.choice(PAIR_STATES) for _ in pairs]
    if all(state == "none" for state in pair_states):
        pair_states[rng.randrange(len(pair_states))] = rng.choice(["first", "second", "both"])

    portal_color_set = set(portal_colors)
    payload_palette = [c for c in range(1, 10) if c != bg and c not in portal_color_set]
    if len(payload_palette) < 2:
        return None

    for (a, b), color in zip(pairs, portal_colors):
        rooms[a]["portal_color"] = int(color)
        rooms[b]["portal_color"] = int(color)

    for room in rooms:
        _carve_portal(
            grid,
            bg,
            int(room["r0"]),
            int(room["c0"]),
            ih,
            iw,
            str(room["portal_side"]),
            int(room["portal_color"]),
            rng,
        )

    both_count = 0
    for pair_idx, ((a, b), state) in enumerate(zip(pairs, pair_states)):
        if state in ("first", "both"):
            _paint_payload(grid, rooms[a], ih, iw, payload_palette, rng)
        if state in ("second", "both"):
            _paint_payload(grid, rooms[b], ih, iw, payload_palette, rng)
        if state == "both":
            both_count += 1

    expected = _expected_from_metadata(grid, bg, ih, iw, rooms, pairs)
    if np.array_equal(grid, expected):
        return None
    if not _bg_is_unique_mode(grid, bg):
        return None
    if np.any(grid == BLACK) and Counter(map(int, grid.flatten()))[BLACK] >= Counter(map(int, grid.flatten()))[bg]:
        return None
    if pair_states.count("none") == len(pair_states) - 1 and both_count == 0 and sum(bool(r["payloads"]) for r in rooms) == 1:
        return None

    produced = main(grid)
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid portal-room sample")
