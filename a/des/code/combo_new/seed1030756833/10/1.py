from __future__ import annotations

from collections import defaultdict
import random
import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)


def _bg_color(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray, *, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR8 if connectivity == 8 else DIR4
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _bbox(coords: np.ndarray | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _in_box(r: int, c: int, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _find_frames(g: np.ndarray, bg: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    by_color: dict[int, list[tuple[int, tuple[int, int, int, int], int]]] = defaultdict(list)
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for comp in _components(g == color, connectivity=4):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 6 or w < 6:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            if not (
                np.all(sub[0, :] == color)
                and np.all(sub[-1, :] == color)
                and np.all(sub[:, 0] == color)
                and np.all(sub[:, -1] == color)
            ):
                continue
            if int(comp.shape[0]) < 2 * h + 2 * w - 4:
                continue
            if not bool(np.any(sub[1:-1, 1:-1] == bg)):
                continue
            by_color[color].append((color, (r0, c0, r1, c1), int(comp.shape[0])))

    if not by_color:
        return []

    def score(color: int) -> tuple[int, int]:
        items = by_color[color]
        return (len(items), sum(area for _col, _box, area in items))

    frame_color = max(by_color, key=score)
    frames = [(color, box) for color, box, _area in by_color[frame_color]]
    frames.sort(key=lambda item: (item[1][0], item[1][1]))
    return frames


def _largest_inner_bg(
    g: np.ndarray, bg: int, box: tuple[int, int, int, int]
) -> np.ndarray | None:
    r0, c0, r1, c1 = box
    inner = g[r0 + 1 : r1, c0 + 1 : c1] == bg
    comps = _components(inner, connectivity=4)
    if not comps:
        return None
    comp = max(comps, key=lambda pts: int(pts.shape[0]))
    shifted = comp.copy()
    shifted[:, 0] += r0 + 1
    shifted[:, 1] += c0 + 1
    return shifted


def _fragment_groups(
    g: np.ndarray,
    bg: int,
    frame_color: int,
    box: tuple[int, int, int, int],
    cavity: np.ndarray,
) -> dict[int, list[tuple[int, int]]]:
    r0, c0, r1, c1 = box
    cav = {(int(r), int(c)) for r, c in cavity}
    seeds_by_color: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            color = int(g[r, c])
            if color in (bg, frame_color):
                continue
            if any((r + dr, c + dc) in cav for dr, dc in DIR8):
                seeds_by_color[color].add((r, c))

    groups: dict[int, list[tuple[int, int]]] = {}
    for color, seeds in seeds_by_color.items():
        local = np.zeros((r1 - r0 - 1, c1 - c0 - 1), dtype=bool)
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if int(g[r, c]) == color:
                    local[r - r0 - 1, c - c0 - 1] = True
        pts: set[tuple[int, int]] = set()
        for comp in _components(local, connectivity=8):
            shifted = {(int(r + r0 + 1), int(c + c0 + 1)) for r, c in comp}
            if shifted & seeds:
                pts |= shifted
        if pts:
            groups[color] = sorted(pts)
    return groups


def _direction(cavity: np.ndarray, all_fragments: list[tuple[int, int]], fragment: list[tuple[int, int]]) -> str:
    all_pts = np.array([(int(r), int(c)) for r, c in cavity] + all_fragments, dtype=float)
    center_r = float(all_pts[:, 0].mean())
    center_c = float(all_pts[:, 1].mean())
    frag_arr = np.array(fragment, dtype=float)
    dr = float(frag_arr[:, 0].mean() - center_r)
    dc = float(frag_arr[:, 1].mean() - center_c)
    if abs(dr) >= abs(dc):
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


def _outside_components_by_color(
    g: np.ndarray, bg: int, frame_boxes: list[tuple[int, int, int, int]]
) -> dict[int, list[np.ndarray]]:
    out: dict[int, list[np.ndarray]] = defaultdict(list)
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for comp in _components(g == color, connectivity=4):
            if all(not any(_in_box(int(r), int(c), box) for box in frame_boxes) for r, c in comp):
                out[color].append(comp)
    return out


def _centered_start(lo: int, hi: int, size: int) -> int:
    return int(lo + ((hi - lo + 1) - size) // 2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide outside monochrome shapes into framed cavities matched by same-colored internal fragments.

    Input structure:
    - A dominant background surrounds several same-colored rectangular frame blocks.
    - Each frame contains one enclosed background cavity with a tiny colored fragment on one side of that cavity.
    - Outside the frames are small monochrome connected shapes whose colors match the internal fragments one-to-one.

    Rule:
    - Find each frame cavity and the non-frame fragment pixels adjacent to it.
    - For every fragment color, erase the matching outside shape of that color.
    - Compare the fragment position with the cavity center; the larger row/column offset gives up, down, left, or right.
    - Translate the matching shape into the cavity from that side, aligning its entering side with the cavity wall while preserving the fragment pixels.

    Key insight: The fragment color says which outside shape to use, while the fragment's side of the cavity says which direction the shape slides in from.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    frames = _find_frames(g, bg)
    frame_boxes = [box for _color, box in frames]
    external = _outside_components_by_color(g, bg, frame_boxes)

    moves: list[tuple[int, np.ndarray, list[tuple[int, int]], tuple[int, int]]] = []
    used_sources: set[tuple[int, tuple[tuple[int, int], ...]]] = set()

    for frame_color, box in frames:
        cavity = _largest_inner_bg(g, bg, box)
        if cavity is None or int(cavity.shape[0]) == 0:
            continue
        groups = _fragment_groups(g, bg, frame_color, box, cavity)
        if not groups:
            continue
        all_fragments = [pt for pts in groups.values() for pt in pts]
        cav_r0, cav_c0, cav_r1, cav_c1 = _bbox(cavity)

        for color, fragment in groups.items():
            choices = [comp for comp in external.get(color, []) if int(comp.shape[0]) == int(cavity.shape[0])]
            if len(choices) != 1:
                continue
            source = choices[0]
            key = (int(color), tuple(sorted((int(r), int(c)) for r, c in source)))
            if key in used_sources:
                continue
            used_sources.add(key)

            sr0, sc0, sr1, sc1 = _bbox(source)
            shape_h = sr1 - sr0 + 1
            shape_w = sc1 - sc0 + 1
            local = [(int(r - sr0), int(c - sc0)) for r, c in source]
            side = _direction(cavity, all_fragments, fragment)
            if side == "up":
                tr = cav_r0
                tc = _centered_start(cav_c0, cav_c1, shape_w)
            elif side == "down":
                tr = cav_r1 - shape_h + 1
                tc = _centered_start(cav_c0, cav_c1, shape_w)
            elif side == "left":
                tr = _centered_start(cav_r0, cav_r1, shape_h)
                tc = cav_c0
            else:
                tr = _centered_start(cav_r0, cav_r1, shape_h)
                tc = cav_c1 - shape_w + 1
            moves.append((int(color), source, local, (int(tr), int(tc))))

    out = g.copy()
    for _color, source, _local, _target in moves:
        for r, c in source:
            out[int(r), int(c)] = bg

    h, w = out.shape
    for color, _source, local, (tr, tc) in moves:
        for rr, cc in local:
            r = tr + int(rr)
            c = tc + int(cc)
            if 0 <= r < h and 0 <= c < w and int(out[r, c]) == bg:
                out[r, c] = int(color)

    return out


# ---------------------------------------------------------------------------
# generate_input
# ---------------------------------------------------------------------------

GEN_TRIES = 500
DIRS = ("up", "down", "left", "right")


def _grow_mask(height: int, width: int, area: int) -> np.ndarray:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_shape_mask() -> np.ndarray | None:
    for _ in range(80):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        min_area = max(5, h + w - 1)
        max_area = max(min_area, h * w - random.randint(1, 3))
        area = random.randint(min_area, max_area)
        mask = _grow_mask(h, w, area)
        if int(mask.sum()) != area:
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if int(mask.sum()) == h * w:
            continue
        return mask
    return None


def _edge_run(mask: np.ndarray, direction: str) -> list[int]:
    if direction == "up":
        return [int(c) for c in np.flatnonzero(mask[0, :])]
    if direction == "down":
        return [int(c) for c in np.flatnonzero(mask[-1, :])]
    if direction == "left":
        return [int(r) for r in np.flatnonzero(mask[:, 0])]
    return [int(r) for r in np.flatnonzero(mask[:, -1])]


def _choose_marker_cells(mask: np.ndarray, direction: str) -> list[tuple[int, int]] | None:
    h, w = mask.shape
    run = _edge_run(mask, direction)
    if not run:
        return None
    center = (w - 1) / 2.0 if direction in ("up", "down") else (h - 1) / 2.0
    run.sort(key=lambda x: abs(float(x) - center))
    wants_two = random.random() < 0.45
    chosen = [run[0]]
    if wants_two:
        pairs = [(a, b) for a in run for b in run if abs(a - b) == 1]
        if pairs:
            chosen = list(min(pairs, key=lambda p: abs(((p[0] + p[1]) / 2.0) - center)))
    chosen = sorted(set(chosen))

    if direction == "up":
        return [(1, 2 + c) for c in chosen]
    if direction == "down":
        return [(h + 2, 2 + c) for c in chosen]
    if direction == "left":
        return [(2 + r, 1) for r in chosen]
    return [(2 + r, w + 2) for r in chosen]


def _frame_patches(
    mask: np.ndarray, direction: str, marker_color: int, wall_color: int, bg: int
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = mask.shape
    marker_cells = _choose_marker_cells(mask, direction)
    if not marker_cells:
        return None
    inp = np.full((h + 4, w + 4), int(wall_color), dtype=int)
    out = inp.copy()
    for r, c in np.argwhere(mask):
        rr = int(r) + 2
        cc = int(c) + 2
        inp[rr, cc] = int(bg)
        out[rr, cc] = int(marker_color)
    for rr, cc in marker_cells:
        inp[int(rr), int(cc)] = int(marker_color)
        out[int(rr), int(cc)] = int(marker_color)
    return inp, out


def _sprite_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _can_place(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = blocked.shape
    r0 = top - pad
    c0 = left - pad
    r1 = top + height + pad
    c1 = left + width + pad
    if r0 < 0 or c0 < 0 or r1 > h or c1 > w:
        return False
    return not bool(blocked[r0:r1, c0:c1].any())


def _reserve(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    blocked[top - pad : top + height + pad, left - pad : left + width + pad] = True


def _paste_solid(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    dst[top : top + h, left : left + w] = patch


def _paste_sprite(dst: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    mask = patch != int(bg)
    sub = dst[top : top + h, left : left + w]
    sub[mask] = patch[mask]


def _place_rect(blocked: np.ndarray, height: int, width: int) -> tuple[int, int] | None:
    h, w = blocked.shape
    for _ in range(200):
        if h - height - 1 <= 1 or w - width - 1 <= 1:
            return None
        r = random.randint(1, h - height - 1)
        c = random.randint(1, w - width - 1)
        if _can_place(blocked, r, c, height, width, pad=1):
            return r, c
    return None


def _mode_is_bg(g: np.ndarray, bg: int) -> bool:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))]) == int(bg)


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(2, 4)
    min_dim = 18 + 4 * (n_frames - 2)
    height = random.randint(min_dim, 30)
    width = random.randint(min_dim, 30)
    bg = random.choice([0, 8])
    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)
    wall_color = colors[0]
    marker_colors = colors[1 : 1 + n_frames]
    if len(marker_colors) < n_frames:
        return None

    directions = list(DIRS)
    random.shuffle(directions)
    directions = directions[:n_frames]

    frame_items = []
    for color, direction in zip(marker_colors, directions):
        mask = _random_shape_mask()
        if mask is None:
            return None
        patches = _frame_patches(mask, direction, int(color), int(wall_color), int(bg))
        if patches is None:
            return None
        sprite = _sprite_patch(mask, int(color), int(bg))
        frame_items.append(
            {
                "color": int(color),
                "direction": direction,
                "mask": mask,
                "frame_in": patches[0],
                "frame_out": patches[1],
                "sprite": sprite,
            }
        )

    inp = np.full((height, width), int(bg), dtype=int)
    expected = np.full((height, width), int(bg), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    random.shuffle(frame_items)
    for item in sorted(frame_items, key=lambda x: int(x["frame_in"].size), reverse=True):
        patch_in = item["frame_in"]
        patch_out = item["frame_out"]
        fh, fw = patch_in.shape
        pos = _place_rect(blocked, fh, fw)
        if pos is None:
            return None
        r, c = pos
        _paste_solid(inp, patch_in, r, c)
        _paste_solid(expected, patch_out, r, c)
        _reserve(blocked, r, c, fh, fw, pad=1)
        item["frame_pos"] = (r, c)

    random.shuffle(frame_items)
    for item in frame_items:
        sprite = item["sprite"]
        sh, sw = sprite.shape
        pos = _place_rect(blocked, sh, sw)
        if pos is None:
            return None
        r, c = pos
        _paste_sprite(inp, sprite, r, c, int(bg))
        _reserve(blocked, r, c, sh, sw, pad=1)

    if not _mode_is_bg(inp, int(bg)):
        return None
    if np.array_equal(inp, expected):
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid framed-cavity slide input")
