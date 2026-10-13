import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
FRAME_COLOR = 5
GEN_RETRIES = 400
PLACE_RETRIES = 250


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_to_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted = coords - np.array([r0, c0], dtype=int)
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask, (r0, c0)


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == w - 1)
    )


def _rotate_point_ccw(point: tuple[int, int], shape: tuple[int, int], k: int) -> tuple[int, int]:
    r, c = map(int, point)
    h, w = map(int, shape)
    for _ in range(k % 4):
        r, c = w - 1 - c, r
        h, w = w, h
    return int(r), int(c)


def _mask_has_internal_hole(mask: np.ndarray) -> bool:
    inv = ~mask
    for comp in _components_from_mask(inv):
        if not _touches_border(comp, mask.shape):
            return True
    return False


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    seen_bboxes: set[tuple[int, int, int, int]] = set()
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for coords in _components_from_mask(g == color):
            r0, c0, r1, c1 = _bbox(coords)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            if not (
                np.all(sub[0, :] == color)
                and np.all(sub[-1, :] == color)
                and np.all(sub[:, 0] == color)
                and np.all(sub[:, -1] == color)
            ):
                continue
            holes = []
            for comp in _components_from_mask(sub == bg):
                if not _touches_border(comp, sub.shape):
                    holes.append(comp)
            if len(holes) != 1:
                continue
            marker_coords = np.argwhere((sub != bg) & (sub != color))
            if len(marker_coords) != 1:
                continue
            mr, mc = map(int, marker_coords[0])
            marker_abs = (r0 + mr, c0 + mc)
            hole = holes[0]
            hole_abs = hole + np.array([r0, c0], dtype=int)
            hole_set = {tuple(map(int, p)) for p in hole_abs}
            target_neighbors = [
                (nr, nc)
                for nr, nc in _neighbors4(*marker_abs)
                if (nr, nc) in hole_set
            ]
            if len(target_neighbors) != 1:
                continue
            hole_mask, (hr0, hc0) = _coords_to_mask(hole_abs)
            entry = {
                "frame_color": int(color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "marker_pos": marker_abs,
                "marker_color": int(g[marker_abs]),
                "hole_coords": hole_abs,
                "hole_mask": hole_mask,
                "hole_origin": (int(hr0), int(hc0)),
                "target_abs": tuple(map(int, target_neighbors[0])),
            }
            if entry["bbox"] in seen_bboxes:
                continue
            seen_bboxes.add(entry["bbox"])
            frames.append(entry)
    frames.sort(key=lambda x: (x["bbox"][0], x["bbox"][1]))
    return frames


def _find_fragments(grid: np.ndarray, bg: int, frame_colors: set[int], marker_positions: set[tuple[int, int]]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    movable = g != bg
    for fc in frame_colors:
        movable &= g != int(fc)
    fragments: list[dict] = []
    for coords in _components_from_mask(movable):
        coord_list = [tuple(map(int, p)) for p in coords]
        if len(coord_list) == 1 and coord_list[0] in marker_positions:
            continue
        colors = [int(g[r, c]) for r, c in coord_list]
        counts = Counter(colors)
        assert len(counts) == 2, "every fragment must use exactly two colors"
        anchor_candidates = [col for col, cnt in counts.items() if cnt == 1]
        assert len(anchor_candidates) == 1, "every fragment must have exactly one singleton anchor color"
        anchor_color = int(anchor_candidates[0])
        body_color = int(next(col for col, cnt in counts.items() if cnt != 1))
        mask, origin = _coords_to_mask(coords)
        anchor_abs = next((r, c) for r, c in coord_list if int(g[r, c]) == anchor_color)
        anchor_rel = (int(anchor_abs[0] - origin[0]), int(anchor_abs[1] - origin[1]))
        fragments.append(
            {
                "coords": coord_list,
                "mask": mask,
                "origin": origin,
                "anchor_color": anchor_color,
                "body_color": body_color,
                "anchor_rel": anchor_rel,
            }
        )
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored fragment to the frame hole with the same marker/anchor color,
    rotate it so the full shape fits and the anchor lands next to the marker,
    fill the hole, and erase all loose fragments.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames = _find_frames(g, bg)
    assert frames, "expected at least one frame"
    frame_colors = {int(frame["frame_color"]) for frame in frames}
    marker_positions = {tuple(frame["marker_pos"]) for frame in frames}
    fragments = _find_fragments(g, bg, frame_colors, marker_positions)

    out = g.copy()
    for frag in fragments:
        for r, c in frag["coords"]:
            out[r, c] = bg

    by_anchor: dict[int, list[dict]] = {}
    for frag in fragments:
        by_anchor.setdefault(int(frag["anchor_color"]), []).append(frag)

    for frame in frames:
        marker_color = int(frame["marker_color"])
        matched = by_anchor.get(marker_color, [])
        assert len(matched) == 1, "each frame must match exactly one fragment by anchor color"
        frag = matched[0]
        target_rel = (
            int(frame["target_abs"][0] - frame["hole_origin"][0]),
            int(frame["target_abs"][1] - frame["hole_origin"][1]),
        )
        valid_rotations = []
        for k in range(4):
            rot_mask = np.rot90(frag["mask"], k)
            rot_anchor = _rotate_point_ccw(frag["anchor_rel"], frag["mask"].shape, k)
            if rot_mask.shape != frame["hole_mask"].shape:
                continue
            if np.array_equal(rot_mask, frame["hole_mask"]) and rot_anchor == target_rel:
                valid_rotations.append((k, rot_mask, rot_anchor))
        assert len(valid_rotations) == 1, "shape fit plus anchor adjacency must determine a unique rotation"
        _k, mask, anchor_rel = valid_rotations[0]
        hr0, hc0 = frame["hole_origin"]
        rr, cc = np.argwhere(mask).T
        out[hr0 + rr, hc0 + cc] = int(frag["body_color"])
        out[hr0 + anchor_rel[0], hc0 + anchor_rel[1]] = int(frag["anchor_color"])

    return out


def _random_connected_mask(rng: random.Random) -> np.ndarray | None:
    for _ in range(80):
        h = rng.randint(2, 5)
        w = rng.randint(2, 5)
        area = rng.randint(3, min(h * w, 8))
        seed = (rng.randrange(h), rng.randrange(w))
        cells = {seed}
        frontier = {p for p in _neighbors4(*seed) if 0 <= p[0] < h and 0 <= p[1] < w}
        while len(cells) < area and frontier:
            pick = rng.choice(tuple(frontier))
            frontier.remove(pick)
            cells.add(pick)
            for np_r, np_c in _neighbors4(*pick):
                if 0 <= np_r < h and 0 <= np_c < w and (np_r, np_c) not in cells:
                    frontier.add((np_r, np_c))
        if len(cells) != area:
            continue
        coords = np.array(sorted(cells), dtype=int)
        mask, _ = _coords_to_mask(coords)
        if mask.sum() < 3:
            continue
        if _mask_has_internal_hole(mask):
            continue
        return mask
    return None


def _rotation_solution_count(
    fragment_mask: np.ndarray,
    fragment_anchor_rel: tuple[int, int],
    hole_mask: np.ndarray,
    target_rel: tuple[int, int],
) -> int:
    count = 0
    for k in range(4):
        rot_mask = np.rot90(fragment_mask, k)
        rot_anchor = _rotate_point_ccw(fragment_anchor_rel, fragment_mask.shape, k)
        if rot_mask.shape == hole_mask.shape and np.array_equal(rot_mask, hole_mask) and rot_anchor == target_rel:
            count += 1
    return count


def _build_frame_and_fragment(
    rng: random.Random,
    bg: int,
    marker_color: int,
    body_color: int,
) -> tuple[dict, dict] | None:
    for _ in range(100):
        hole_mask = _random_connected_mask(rng)
        if hole_mask is None:
            continue
        hh, hw = map(int, hole_mask.shape)
        pad_top = rng.randint(2, 4)
        pad_bottom = rng.randint(2, 4)
        pad_left = rng.randint(2, 4)
        pad_right = rng.randint(2, 4)
        fh = pad_top + hh + pad_bottom
        fw = pad_left + hw + pad_right
        frame_patch = np.full((fh, fw), FRAME_COLOR, dtype=int)
        hole_slice = frame_patch[pad_top : pad_top + hh, pad_left : pad_left + hw]
        hole_slice[hole_mask] = bg

        candidates = []
        for r in range(1, fh - 1):
            for c in range(1, fw - 1):
                if int(frame_patch[r, c]) != FRAME_COLOR:
                    continue
                hole_neighbors = [
                    (nr, nc)
                    for nr, nc in _neighbors4(r, c)
                    if 0 <= nr < fh and 0 <= nc < fw and int(frame_patch[nr, nc]) == bg
                ]
                if len(hole_neighbors) == 1:
                    candidates.append((r, c, hole_neighbors[0]))
        if not candidates:
            continue
        mr, mc, target_local = rng.choice(candidates)
        frame_patch[mr, mc] = marker_color
        if len(_components_from_mask(frame_patch == FRAME_COLOR)) != 1:
            continue

        target_rel = (int(target_local[0] - pad_top), int(target_local[1] - pad_left))
        input_rot = rng.randrange(4)
        fragment_mask = np.rot90(hole_mask, input_rot)
        fragment_anchor_rel = _rotate_point_ccw(target_rel, hole_mask.shape, input_rot)
        if _rotation_solution_count(fragment_mask, fragment_anchor_rel, hole_mask, target_rel) != 1:
            continue

        fragment_patch = np.full(fragment_mask.shape, bg, dtype=int)
        fragment_patch[fragment_mask] = body_color
        fragment_patch[fragment_anchor_rel] = marker_color

        frame_info = {
            "patch": frame_patch,
            "hole_mask": hole_mask,
            "hole_offset": (pad_top, pad_left),
            "target_rel": target_rel,
            "marker_color": int(marker_color),
            "body_color": int(body_color),
        }
        fragment_info = {
            "patch": fragment_patch,
            "mask": fragment_mask,
            "anchor_rel": fragment_anchor_rel,
            "anchor_color": int(marker_color),
            "body_color": int(body_color),
        }
        return frame_info, fragment_info
    return None


def _build_distractor_fragment(
    rng: random.Random,
    bg: int,
    anchor_color: int,
    body_color: int,
) -> dict | None:
    for _ in range(80):
        mask = _random_connected_mask(rng)
        if mask is None:
            continue
        coords = np.argwhere(mask)
        anchor_rel = tuple(map(int, coords[rng.randrange(len(coords))]))
        patch = np.full(mask.shape, bg, dtype=int)
        patch[mask] = body_color
        patch[anchor_rel] = anchor_color
        return {
            "patch": patch,
            "mask": mask,
            "anchor_rel": anchor_rel,
            "anchor_color": int(anchor_color),
            "body_color": int(body_color),
        }
    return None


def _can_place(reserved: np.ndarray, top: int, left: int, shape: tuple[int, int], gap: int) -> bool:
    h, w = map(int, shape)
    gh, gw = reserved.shape
    if top < 0 or left < 0 or top + h > gh or left + w > gw:
        return False
    r0 = max(0, top - gap)
    c0 = max(0, left - gap)
    r1 = min(gh, top + h + gap)
    c1 = min(gw, left + w + gap)
    return not bool(reserved[r0:r1, c0:c1].any())


def _place_patch(grid: np.ndarray, top: int, left: int, patch: np.ndarray):
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg_choices = [c for c in range(10) if c != FRAME_COLOR]
    bg = int(rng.choice(bg_choices))
    pair_count = rng.randint(1, 3)
    distractor_count = rng.randint(0, 2)

    color_pool = [c for c in range(10) if c not in {bg, FRAME_COLOR}]
    rng.shuffle(color_pool)
    if pair_count + distractor_count > len(color_pool):
        return None
    marker_colors = color_pool[:pair_count]
    distractor_anchor_colors = color_pool[pair_count : pair_count + distractor_count]
    body_pool = [c for c in range(10) if c not in {bg, FRAME_COLOR}]

    frames = []
    fragments = []
    expected_fill = []

    for marker_color in marker_colors:
        body_candidates = [c for c in body_pool if c != marker_color]
        body_color = int(rng.choice(body_candidates))
        built = _build_frame_and_fragment(rng, bg, marker_color, body_color)
        if built is None:
            return None
        frame_info, fragment_info = built
        frames.append(frame_info)
        fragments.append({"kind": "matched", **fragment_info})
        expected_fill.append(
            {
                "hole_mask": frame_info["hole_mask"],
                "hole_offset": frame_info["hole_offset"],
                "target_rel": frame_info["target_rel"],
                "anchor_color": marker_color,
                "body_color": body_color,
            }
        )

    for anchor_color in distractor_anchor_colors:
        body_candidates = [c for c in body_pool if c != anchor_color]
        body_color = int(rng.choice(body_candidates))
        distractor = _build_distractor_fragment(rng, bg, anchor_color, body_color)
        if distractor is None:
            return None
        fragments.append({"kind": "distractor", **distractor})

    min_side = max(18, 12 + 4 * pair_count + 2 * distractor_count)
    max_side = max(min_side, min(30, min_side + 6))
    gh = rng.randint(min_side, max_side)
    gw = rng.randint(min_side, max_side)
    grid = np.full((gh, gw), bg, dtype=int)
    reserved = np.zeros((gh, gw), dtype=bool)

    frame_order = sorted(range(len(frames)), key=lambda i: frames[i]["patch"].shape[0] * frames[i]["patch"].shape[1], reverse=True)
    frame_positions: dict[int, tuple[int, int]] = {}
    for idx in frame_order:
        patch = frames[idx]["patch"]
        h, w = patch.shape
        placed = False
        for _ in range(PLACE_RETRIES):
            top = rng.randint(0, gh - h)
            left = rng.randint(0, gw - w)
            if not _can_place(reserved, top, left, patch.shape, gap=1):
                continue
            _place_patch(grid, top, left, patch)
            reserved[top : top + h, left : left + w] = True
            frame_positions[idx] = (top, left)
            placed = True
            break
        if not placed:
            return None

    fragment_order = sorted(range(len(fragments)), key=lambda i: fragments[i]["patch"].shape[0] * fragments[i]["patch"].shape[1], reverse=True)
    fragment_positions: dict[int, tuple[int, int]] = {}
    for idx in fragment_order:
        patch = fragments[idx]["patch"]
        h, w = patch.shape
        placed = False
        for _ in range(PLACE_RETRIES):
            top = rng.randint(0, gh - h)
            left = rng.randint(0, gw - w)
            if not _can_place(reserved, top, left, patch.shape, gap=1):
                continue
            _place_patch(grid, top, left, patch)
            reserved[top : top + h, left : left + w] = True
            fragment_positions[idx] = (top, left)
            placed = True
            break
        if not placed:
            return None

    if _get_bg(grid) != bg:
        return None

    expected = grid.copy()
    for idx, frag in enumerate(fragments):
        top, left = fragment_positions[idx]
        mask = frag["mask"]
        expected[top : top + mask.shape[0], left : left + mask.shape[1]][mask] = bg

    for idx, fill in enumerate(expected_fill):
        frame_top, frame_left = frame_positions[idx]
        hole_top = frame_top + int(fill["hole_offset"][0])
        hole_left = frame_left + int(fill["hole_offset"][1])
        hole_mask = fill["hole_mask"]
        expected[hole_top : hole_top + hole_mask.shape[0], hole_left : hole_left + hole_mask.shape[1]][hole_mask] = int(fill["body_color"])
        tr, tc = fill["target_rel"]
        expected[hole_top + tr, hole_left + tc] = int(fill["anchor_color"])

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None

    frames_found = _find_frames(grid, bg)
    if len(frames_found) != pair_count:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
