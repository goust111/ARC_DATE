import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def components_mask(mask: np.ndarray, conn8: bool = False) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR8 if conn8 else DIR4
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def bbox_of(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def normalize_shape(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], frozenset[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox_of(coords)
    rel = frozenset((r - r0, c - c0) for r, c in coords)
    return (r1 - r0 + 1, c1 - c0 + 1), rel


def crop_component(grid: np.ndarray, coords: list[tuple[int, int]], fill: int = 0) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = bbox_of(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(fill), dtype=int)
    mask = np.zeros_like(patch, dtype=bool)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
        mask[r - r0, c - c0] = True
    return patch, mask, (r0, c0, r1, c1)


def dihedral_variants(patch: np.ndarray, mask: np.ndarray):
    seen: set[tuple[bytes, bytes, tuple[int, int]]] = set()
    for flip in (False, True):
        p0 = np.fliplr(patch) if flip else patch
        m0 = np.fliplr(mask) if flip else mask
        for k in range(4):
            p = np.rot90(p0, k)
            m = np.rot90(m0, k)
            key = (p.tobytes(), m.tobytes(), p.shape)
            if key in seen:
                continue
            seen.add(key)
            yield p.copy(), m.copy(), bool(flip)


def swap_two_colors(patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = [int(v) for v in patch[mask]]
    colors = sorted(set(vals))
    if len(colors) != 2:
        return patch
    a, b = colors
    out = patch.copy()
    out[(patch == a) & mask] = b
    out[(patch == b) & mask] = a
    return out


def frame_cavity(frame_mask: np.ndarray) -> list[tuple[int, int]]:
    comps = components_mask(~frame_mask, conn8=False)
    assert comps, "frame must contain a cavity in its bbox"
    comps.sort(key=len, reverse=True)
    return comps[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered two-color sprites with border-touching frames by cavity area,
    rotate/flip sprites so one color region fills the cavity, and swap the two
    sprite colors when the chosen docking transform uses reflection.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    # Border-touching monochrome 4-connected components are fixed frames.
    frame_entries = []
    frame_pixels = np.zeros((h, w), dtype=bool)
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        comps = components_mask(g == int(color), conn8=False)
        for comp in comps:
            if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in comp):
                patch, mask, bbox = crop_component(g, comp, fill=bg)
                cavity = frame_cavity(mask)
                cav_shape = normalize_shape(cavity)
                frame_entries.append(
                    {
                        "color": int(color),
                        "coords": comp,
                        "bbox": bbox,
                        "mask": mask,
                        "cavity_coords": cavity,
                        "cavity_shape": cav_shape,
                        "cavity_area": len(cavity),
                    }
                )
                for r, c in comp:
                    frame_pixels[r, c] = True

    assert frame_entries, "expected at least one border-touching frame"

    # Remaining non-background pixels form movable 2-color sprites.
    movable_mask = (g != int(bg)) & (~frame_pixels)
    sprite_entries = []
    for comp in components_mask(movable_mask, conn8=True):
        patch, mask, bbox = crop_component(g, comp, fill=bg)
        vals = [int(v) for v in patch[mask]]
        counts = Counter(vals)
        assert len(counts) == 2, "every movable sprite must use exactly two colors"
        sprite_entries.append({"coords": comp, "bbox": bbox, "patch": patch, "mask": mask, "counts": counts})

    assert len(sprite_entries) == len(frame_entries), "generator promises one sprite per frame"

    area_to_frame = {entry["cavity_area"]: entry for entry in frame_entries}
    assert len(area_to_frame) == len(frame_entries), "cavity areas must be unique"

    out = np.full_like(g, int(bg))
    for entry in frame_entries:
        for r, c in entry["coords"]:
            out[r, c] = entry["color"]

    for sprite in sprite_entries:
        plug_candidates = [color for color, cnt in sprite["counts"].items() if int(cnt) in area_to_frame]
        assert len(plug_candidates) == 1, "pairing by area must be unambiguous"
        plug_color = int(plug_candidates[0])
        frame = area_to_frame[int(sprite["counts"][plug_color])]
        cav_h, cav_w = frame["cavity_shape"][0]
        cavity_rel = frame["cavity_shape"][1]
        fr0, fc0, _fr1, _fc1 = frame["bbox"]

        chosen = None
        cav_rmin = min(r for r, _ in frame["cavity_coords"])
        cav_cmin = min(c for _, c in frame["cavity_coords"])
        for patch_v, mask_v, used_flip in dihedral_variants(sprite["patch"], sprite["mask"]):
            plug_coords = [(int(r), int(c)) for r, c in np.argwhere((patch_v == int(plug_color)) & mask_v)]
            if not plug_coords:
                continue
            plug_shape = normalize_shape(plug_coords)
            if plug_shape != ((cav_h, cav_w), cavity_rel):
                continue

            pr0, pc0, _pr1, _pc1 = bbox_of(plug_coords)
            dr = int(fr0 + cav_rmin - pr0)
            dc = int(fc0 + cav_cmin - pc0)

            ok = True
            translated: list[tuple[int, int, int, bool]] = []
            cavity_abs = {(fr0 + r, fc0 + c) for r, c in frame["cavity_coords"]}
            for rr, cc in np.argwhere(mask_v):
                ar = int(fr0 + rr + cav_rmin - pr0)
                ac = int(fc0 + cc + cav_cmin - pc0)
                if not (0 <= ar < h and 0 <= ac < w):
                    ok = False
                    break
                is_plug = int(patch_v[rr, cc]) == int(plug_color)
                if is_plug:
                    if (ar, ac) not in cavity_abs:
                        ok = False
                        break
                else:
                    if out[ar, ac] != int(bg):
                        ok = False
                        break
                translated.append((ar, ac, int(patch_v[rr, cc]), is_plug))
            if not ok:
                continue

            plug_set = {(ar, ac) for ar, ac, _col, is_plug in translated if is_plug}
            if plug_set != cavity_abs:
                continue
            chosen = (patch_v, mask_v, used_flip, translated, dr, dc)
            break

        assert chosen is not None, "must find a docking transform"
        patch_v, mask_v, used_flip, translated, dr, dc = chosen
        render = swap_two_colors(patch_v, mask_v) if used_flip else patch_v
        for ar, ac, _orig_col, _is_plug in translated:
            rr = ar - dr
            cc = ac - dc
            out[ar, ac] = int(render[rr, cc])

    return out


GRID_H_RANGE = (18, 27)
GRID_W_RANGE = (18, 27)
PAIR_COUNT_CHOICES = [1, 2, 3]
MAX_GEN_TRIES = 800


def random_connected_mask(height: int, width: int, area: int, must_touch_side: str) -> np.ndarray:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return random_connected_mask(height, width, area, must_touch_side)
        cells.add(random.choice(frontier))

    def touches_side(current: set[tuple[int, int]]) -> bool:
        if must_touch_side == "right":
            return any(c == width - 1 for _r, c in current)
        if must_touch_side == "left":
            return any(c == 0 for _r, c in current)
        if must_touch_side == "top":
            return any(r == 0 for r, _c in current)
        return any(r == height - 1 for r, _c in current)

    if not touches_side(cells):
        pivot = random.choice(list(cells))
        if must_touch_side == "right":
            cells.remove(pivot)
            cells.add((pivot[0], width - 1))
        elif must_touch_side == "left":
            cells.remove(pivot)
            cells.add((pivot[0], 0))
        elif must_touch_side == "top":
            cells.remove(pivot)
            cells.add((0, pivot[1]))
        else:
            cells.remove(pivot)
            cells.add((height - 1, pivot[1]))
        if len(components_mask(np.array([[((r, c) in cells) for c in range(width)] for r in range(height)], dtype=bool))) != 1:
            return random_connected_mask(height, width, area, must_touch_side)

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def random_outside_mask(hole_mask: np.ndarray, extra_area: int) -> np.ndarray | None:
    h, w = hole_mask.shape
    seeds = [(r, w) for r in range(h) if hole_mask[r, w - 1]]
    if not seeds:
        return None
    cells = {random.choice(seeds)}
    while len(cells) < extra_area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and w <= nc < w + 3 and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    mask = np.zeros((h, w + 3), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def transform_mask(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return mask
    if side == "top":
        return np.rot90(mask, 3)
    if side == "right":
        return np.rot90(mask, 2)
    return np.rot90(mask, 1)


def transform_patch(patch: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return patch
    if side == "top":
        return np.rot90(patch, 3)
    if side == "right":
        return np.rot90(patch, 2)
    return np.rot90(patch, 1)


def place_patch(canvas: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> bool:
    h, w = patch.shape
    if top < 0 or left < 0 or top + h > canvas.shape[0] or left + w > canvas.shape[1]:
        return False
    view = canvas[top : top + h, left : left + w]
    mask = patch != int(bg)
    if np.any(mask & (view != int(bg))):
        return False
    view[mask] = patch[mask]
    return True


def inverse_variant_from_output(output_patch: np.ndarray, flip: bool, rot_k: int) -> np.ndarray:
    patch = output_patch.copy()
    if flip:
        patch = swap_two_colors(patch, patch != 0)
    patch = np.rot90(patch, -rot_k)
    if flip:
        patch = np.fliplr(patch)
    return patch


def build_pair(area: int, frame_color: int, plug_color: int, outside_color: int, side: str, use_flip: bool):
    for _ in range(80):
        hole_h = random.randint(2, 4)
        hole_w = random.randint(2, 4)
        if area > hole_h * hole_w:
            continue
        hole = random_connected_mask(hole_h, hole_w, area, "right")
        outside_area = random.choice([2, 3])
        outside = random_outside_mask(hole, outside_area)
        if outside is None:
            continue

        frame = np.full((hole_h + 2, hole_w + 1), int(frame_color), dtype=int)
        frame[1 : 1 + hole_h, 1 : 1 + hole_w][hole] = 0

        sprite = np.zeros((hole_h, hole_w + 3), dtype=int)
        sprite[:, :hole_w][hole] = int(plug_color)
        sprite[outside] = int(outside_color)

        frame = transform_patch(frame, side)
        sprite_out = transform_patch(sprite, side)

        flip_flag = bool(use_flip)
        rot_k = random.randrange(4)
        input_sprite = inverse_variant_from_output(sprite_out, flip_flag, rot_k)
        if flip_flag:
            # Make sure the solving transform truly needs a reflection.
            pass

        return frame, sprite_out, input_sprite, plug_color, flip_flag, rot_k
    return None


def dock_position(frame_patch: np.ndarray, sprite_patch: np.ndarray, plug_color: int) -> tuple[int, int] | None:
    frame_mask = frame_patch != 0
    cavity = frame_cavity(frame_mask)
    cav_r0, cav_c0, _cav_r1, _cav_c1 = bbox_of(cavity)
    plug_coords = [(int(r), int(c)) for r, c in np.argwhere(sprite_patch == int(plug_color))]
    if not plug_coords:
        return None
    plug_r0, plug_c0, _plug_r1, _plug_c1 = bbox_of(plug_coords)
    if normalize_shape(cavity) != normalize_shape(plug_coords):
        return None
    return cav_r0 - plug_r0, cav_c0 - plug_c0


def pair_bbox_for_side(frame_shape: tuple[int, int], sprite_shape: tuple[int, int], side: str) -> tuple[int, int]:
    fh, fw = frame_shape
    sh, sw = sprite_shape
    if side in ("left", "right"):
        return max(fh, sh), fw + sw
    return fh + sh, max(fw, sw)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_TRIES):
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        grid = np.zeros((h, w), dtype=int)
        expected = np.zeros((h, w), dtype=int)

        pair_count = random.choice(PAIR_COUNT_CHOICES)
        sides = random.sample(["left", "top", "right", "bottom"], k=pair_count)
        colors = list(range(1, 10))
        random.shuffle(colors)
        if len(colors) < pair_count * 3:
            continue

        areas_pool = list(range(4, 9))
        random.shuffle(areas_pool)
        areas = areas_pool[:pair_count]

        pair_specs = []
        ok = True
        for i in range(pair_count):
            frame_color, plug_color, outside_color = colors[3 * i : 3 * i + 3]
            side = sides[i]
            use_flip = (i + random.randrange(2)) % 2 == 0
            built = build_pair(areas[i], frame_color, plug_color, outside_color, side, use_flip)
            if built is None:
                ok = False
                break
            frame_patch, sprite_out, sprite_in, plug_color, _flip_flag, _rot_k = built
            pair_specs.append(
                {
                    "side": side,
                    "frame_patch": frame_patch,
                    "sprite_out": sprite_out,
                    "sprite_in": sprite_in,
                    "plug_color": plug_color,
                }
            )
        if not ok:
            continue

        used_spans = {"left": [], "right": [], "top": [], "bottom": []}
        placed_frames = []
        for spec in pair_specs:
            fh, fw = spec["frame_patch"].shape
            side = spec["side"]
            placed = False
            for _try in range(80):
                if side == "left":
                    top = random.randint(1, h - fh - 1)
                    left = 0
                    span = (top - 1, top + fh)
                elif side == "right":
                    top = random.randint(1, h - fh - 1)
                    left = w - fw
                    span = (top - 1, top + fh)
                elif side == "top":
                    top = 0
                    left = random.randint(1, w - fw - 1)
                    span = (left - 1, left + fw)
                else:
                    top = h - fh
                    left = random.randint(1, w - fw - 1)
                    span = (left - 1, left + fw)

                if any(not (span[1] < a or b < span[0]) for a, b in used_spans[side]):
                    continue
                temp = grid.copy()
                if not place_patch(temp, spec["frame_patch"], top, left, 0):
                    continue
                grid = temp
                place_patch(expected, spec["frame_patch"], top, left, 0)
                used_spans[side].append(span)
                placed_frames.append((top, left))
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        # Rebuild expected cleanly after frame placement.
        expected.fill(0)
        for spec, (top, left) in zip(pair_specs, placed_frames):
            place_patch(expected, spec["frame_patch"], top, left, 0)

        for spec, (top, left) in zip(pair_specs, placed_frames):
            dock = dock_position(spec["frame_patch"], spec["sprite_out"], spec["plug_color"])
            if dock is None:
                ok = False
                break
            out_top = top + dock[0]
            out_left = left + dock[1]
            if not place_patch(expected, spec["sprite_out"], out_top, out_left, 0):
                ok = False
                break
        if not ok:
            continue

        for spec in pair_specs:
            placed = False
            ih, iw = spec["sprite_in"].shape
            for _try in range(120):
                top = random.randint(2, h - ih - 3)
                left = random.randint(2, w - iw - 3)
                temp = grid.copy()
                if not place_patch(temp, spec["sprite_in"], top, left, 0):
                    continue
                # Keep sprites away from border and away from frames for readability.
                bbox_view = temp[max(0, top - 1) : min(h, top + ih + 1), max(0, left - 1) : min(w, left + iw + 1)]
                if np.count_nonzero(bbox_view) != np.count_nonzero(spec["sprite_in"]):
                    continue
                grid = temp
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        if len(components_mask((grid != 0), conn8=True)) < pair_count * 2:
            continue
        try:
            solved = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(solved, expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
