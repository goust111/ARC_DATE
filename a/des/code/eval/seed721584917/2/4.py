import random
from collections import Counter

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def rect_path(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        path.append((r0, c))
    for r in range(r0 + 1, r1):
        path.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        path.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        path.append((r, c0))
    return path


def _find_tracks(grid: np.ndarray, bg: int) -> list[dict]:
    tracks: list[dict] = []
    for comp in components_mask(grid != bg):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        path = rect_path(r0, c0, r1, c1)
        if set(comp) != set(path):
            continue
        colors = Counter(int(grid[r, c]) for r, c in comp)
        track_color, track_count = colors.most_common(1)[0]
        if track_count * 2 <= len(path):
            continue
        markers: dict[int, int] = {}
        for idx, (r, c) in enumerate(path):
            cell = int(grid[r, c])
            if cell != track_color:
                markers[idx] = cell
        tracks.append(
            {
                "track_color": int(track_color),
                "path": path,
                "markers": markers,
            }
        )
    return tracks


def _decode_arrow(comp: list[tuple[int, int]]) -> dict | None:
    pts = set(comp)
    if len(pts) < 3:
        return None

    neigh: dict[tuple[int, int], list[tuple[int, int]]] = {}
    degree: dict[tuple[int, int], int] = {}
    edges = 0
    for p in pts:
        cur = [q for q in neighbors4(*p) if q in pts]
        neigh[p] = cur
        degree[p] = len(cur)
        edges += len(cur)
    if edges // 2 != len(pts) - 1:
        return None
    if sum(d == 1 for d in degree.values()) != 2:
        return None

    bend = None
    for p, cur in neigh.items():
        if len(cur) != 2:
            continue
        (r1, c1), (r2, c2) = cur
        if r1 != r2 and c1 != c2:
            if bend is not None:
                return None
            bend = p
    if bend is None:
        return None

    arms = []
    for first in neigh[bend]:
        prev = bend
        cur = first
        length = 1
        while degree[cur] != 1:
            nxt = [q for q in neigh[cur] if q != prev]
            if len(nxt) != 1:
                return None
            prev, cur = cur, nxt[0]
            length += 1
        arms.append((length, first, cur))
    if len(arms) != 2:
        return None
    arms.sort(key=lambda item: item[0])
    short_len, short_first, short_end = arms[0]
    long_len, long_first, _long_end = arms[1]
    if short_len != 1 or long_len < 1:
        return None

    shaft_dir = (bend[0] - long_first[0], bend[1] - long_first[1])
    branch_dir = (short_end[0] - bend[0], short_end[1] - bend[1])
    cross = shaft_dir[0] * branch_dir[1] - shaft_dir[1] * branch_dir[0]
    if cross == 0:
        return None
    return {"clockwise": cross > 0, "steps": long_len + 1}


def _find_arrows(grid: np.ndarray, tracks: list[dict]) -> dict[int, dict]:
    track_mask = np.zeros(grid.shape, dtype=bool)
    track_colors = set()
    for track in tracks:
        track_colors.add(track["track_color"])
        for r, c in track["path"]:
            track_mask[r, c] = True

    arrows: dict[int, dict] = {}
    for color in sorted(track_colors):
        comps = components_mask((grid == color) & (~track_mask))
        decoded = [_decode_arrow(comp) for comp in comps]
        valid = [item for item in decoded if item is not None]
        if valid:
            arrows[color] = valid[0]
    return arrows


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    tracks = _find_tracks(g, bg)
    arrows = _find_arrows(g, tracks)

    out = g.copy()
    for track in tracks:
        arrow = arrows.get(track["track_color"])
        if arrow is None:
            continue
        path = track["path"]
        delta = int(arrow["steps"]) if arrow["clockwise"] else -int(arrow["steps"])
        for r, c in path:
            out[r, c] = track["track_color"]
        for idx, color in track["markers"].items():
            nr, nc = path[(idx + delta) % len(path)]
            out[nr, nc] = color
    return out


GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
TRACK_COUNT_CHOICES = (1, 2, 3, 4)
TRACK_H_RANGE = (4, 9)
TRACK_W_RANGE = (4, 11)
MAX_GEN_RETRIES = 400
MAX_PLACE_RETRIES = 120


def _sample_marker_indices(path_len: int, count: int) -> list[int] | None:
    if count == 1:
        return [random.randrange(path_len)]
    candidates = list(range(path_len))
    for _ in range(100):
        picks = sorted(random.sample(candidates, count))
        ok = True
        for i in range(count):
            a = picks[i]
            b = picks[(i + 1) % count]
            dist = (b - a) % path_len
            if dist <= 1:
                ok = False
                break
        if ok:
            return picks
    return None


def _build_arrow_local(
    track_h: int,
    track_w: int,
    orientation: str,
    side: str,
    steps: int,
    clockwise: bool,
) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    if orientation == "horizontal":
        start_c = random.randint(0, track_w - steps)
        end_c = start_c + steps - 1
        if side == "top":
            shaft_r = -2 if clockwise else -3
            branch_r = -3 if clockwise else -2
        else:
            shaft_r = track_h + 2 if clockwise else track_h + 1
            branch_r = track_h + 1 if clockwise else track_h + 2
        for c in range(start_c, start_c + steps):
            coords.append((shaft_r, c))
        coords.append((branch_r, end_c))
        return coords

    start_r = random.randint(0, track_h - steps)
    end_r = start_r + steps - 1
    if side == "left":
        shaft_c = -3 if clockwise else -2
        branch_c = -2 if clockwise else -3
    else:
        shaft_c = track_w + 1 if clockwise else track_w + 2
        branch_c = track_w + 2 if clockwise else track_w + 1
    for r in range(start_r, start_r + steps):
        coords.append((r, shaft_c))
    coords.append((end_r, branch_c))
    return coords


def _make_object(bg: int, track_color: int) -> dict | None:
    track_h = random.randint(*TRACK_H_RANGE)
    track_w = random.randint(*TRACK_W_RANGE)
    path = rect_path(0, 0, track_h - 1, track_w - 1)
    path_len = len(path)

    orientation = random.choice(["horizontal", "vertical"])
    side = random.choice(["top", "bottom"] if orientation == "horizontal" else ["left", "right"])
    steps = random.randint(2, track_w if orientation == "horizontal" else track_h)
    clockwise = bool(random.getrandbits(1))

    max_markers = min(4, max(1, (path_len - 1) // 3))
    marker_count = random.randint(1, max_markers)
    marker_indices = _sample_marker_indices(path_len, marker_count)
    if marker_indices is None:
        return None

    marker_palette = [c for c in range(10) if c != bg and c != track_color]
    if not marker_palette:
        return None
    random.shuffle(marker_palette)
    marker_colors = [marker_palette[i % len(marker_palette)] for i in range(marker_count)]
    output_markers = dict(zip(marker_indices, marker_colors))

    delta = steps if clockwise else -steps
    input_markers = {((idx - delta) % path_len): color for idx, color in output_markers.items()}

    arrow = _build_arrow_local(track_h, track_w, orientation, side, steps, clockwise)
    all_coords = path + arrow
    min_r = min(r for r, _ in all_coords)
    min_c = min(c for _, c in all_coords)
    max_r = max(r for r, _ in all_coords)
    max_c = max(c for _, c in all_coords)

    shift_r = -min_r
    shift_c = -min_c
    local_path = [(r + shift_r, c + shift_c) for r, c in path]
    local_arrow = [(r + shift_r, c + shift_c) for r, c in arrow]

    return {
        "patch_h": max_r - min_r + 1,
        "patch_w": max_c - min_c + 1,
        "track_color": track_color,
        "path": local_path,
        "arrow": local_arrow,
        "input_markers": input_markers,
        "output_markers": output_markers,
    }


def _place_objects(h: int, w: int, objects: list[dict]) -> list[tuple[int, int]] | None:
    order = sorted(range(len(objects)), key=lambda i: objects[i]["patch_h"] * objects[i]["patch_w"], reverse=True)
    reserved = np.zeros((h, w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(objects)

    for idx in order:
        obj = objects[idx]
        ph, pw = obj["patch_h"], obj["patch_w"]
        if ph > h or pw > w:
            return None
        coords = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        random.shuffle(coords)
        placed = False
        for r0, c0 in coords[:MAX_PLACE_RETRIES]:
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + ph + 1)
            cc1 = min(w, c0 + pw + 1)
            if reserved[rr0:rr1, cc0:cc1].any():
                continue
            reserved[rr0:rr1, cc0:cc1] = True
            positions[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None
    return [pos for pos in positions if pos is not None]


def _render_scene(h: int, w: int, bg: int, objects: list[dict], positions: list[tuple[int, int]], use_output: bool) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for obj, (r0, c0) in zip(objects, positions):
        for r, c in obj["path"]:
            grid[r0 + r, c0 + c] = obj["track_color"]
        for r, c in obj["arrow"]:
            grid[r0 + r, c0 + c] = obj["track_color"]
        markers = obj["output_markers"] if use_output else obj["input_markers"]
        for idx, color in markers.items():
            mr, mc = obj["path"][idx]
            grid[r0 + mr, c0 + mc] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    bg = random.randrange(10)
    track_count = random.choice(TRACK_COUNT_CHOICES)
    track_colors = [c for c in range(10) if c != bg]
    random.shuffle(track_colors)
    track_colors = track_colors[:track_count]

    objects: list[dict] = []
    for color in track_colors:
        obj = _make_object(bg, color)
        if obj is None:
            return None
        objects.append(obj)

    min_h = max(max(obj["patch_h"] for obj in objects) + 2, GRID_H_RANGE[0])
    min_w = max(max(obj["patch_w"] for obj in objects) + 2, GRID_W_RANGE[0])
    if min_h >= GRID_H_RANGE[1] or min_w >= GRID_W_RANGE[1]:
        return None

    total_patch_area = sum((obj["patch_h"] + 2) * (obj["patch_w"] + 2) for obj in objects)
    for _ in range(60):
        h = random.randint(min_h, GRID_H_RANGE[1] - 1)
        w = random.randint(min_w, GRID_W_RANGE[1] - 1)
        if h * w < total_patch_area:
            continue
        positions = _place_objects(h, w, objects)
        if positions is None:
            continue

        inp = _render_scene(h, w, bg, objects, positions, use_output=False)
        out = _render_scene(h, w, bg, objects, positions, use_output=True)
        if np.array_equal(inp, out):
            continue
        if get_bg(inp) != bg:
            continue
        if not np.array_equal(main(inp), out):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
