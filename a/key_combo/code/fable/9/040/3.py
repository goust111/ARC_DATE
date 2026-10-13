import random
from collections import Counter, deque

import numpy as np


CORNER_SIGNS = {
    "tl": (-1, -1),
    "tr": (-1, 1),
    "bl": (1, -1),
    "br": (1, 1),
}
CORNER_NAMES = tuple(CORNER_SIGNS)
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        steps = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )

    out = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _find_frames(marker_coords: list[tuple[int, int]]) -> list[dict]:
    coord_set = set(marker_coords)
    rows = sorted({r for r, _ in marker_coords})
    cols = sorted({c for _, c in marker_coords})
    frames = []

    for r0 in rows:
        for r1 in rows:
            side = r1 - r0
            if side <= 0:
                continue
            for c0 in cols:
                c1 = c0 + side
                if c1 not in cols:
                    continue
                corners = {
                    "tl": (r0, c0),
                    "tr": (r0, c1),
                    "bl": (r1, c0),
                    "br": (r1, c1),
                }
                present = {name: pos for name, pos in corners.items() if pos in coord_set}
                if len(present) not in (3, 4):
                    continue
                inside = {
                    pos
                    for pos in coord_set
                    if r0 <= pos[0] <= r1 and c0 <= pos[1] <= c1
                }
                if inside != set(present.values()):
                    continue
                missing_kind = None
                if len(present) == 3:
                    missing_kind = next(name for name in CORNER_NAMES if name not in present)
                frames.append(
                    {
                        "bbox": (r0, c0, r1, c1),
                        "side": side,
                        "present": tuple(sorted(present.values())),
                        "missing_kind": missing_kind,
                        "kind": "full" if len(present) == 4 else "missing",
                    }
                )
    return frames


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _select_frames(marker_coords: list[tuple[int, int]]) -> list[dict]:
    candidates = _find_frames(marker_coords)
    all_markers = set(marker_coords)
    marker_to_candidates = {coord: [] for coord in all_markers}
    for idx, frame in enumerate(candidates):
        for coord in frame["present"]:
            marker_to_candidates[coord].append(idx)

    solutions: list[list[int]] = []

    def search(unused: set[tuple[int, int]], chosen: list[int]) -> None:
        if len(solutions) > 1:
            return
        if not unused:
            solutions.append(chosen.copy())
            return

        coord = min(unused, key=lambda p: len(marker_to_candidates[p]))
        for idx in marker_to_candidates[coord]:
            frame = candidates[idx]
            present = set(frame["present"])
            if not present <= unused:
                continue
            if any(_bboxes_overlap(frame["bbox"], candidates[other]["bbox"]) for other in chosen):
                continue
            chosen.append(idx)
            search(unused - present, chosen)
            chosen.pop()

    search(all_markers, [])
    assert len(solutions) == 1, "expected a unique frame decomposition"
    return [candidates[idx] for idx in solutions[0]]


def _find_marker_color(grid: np.ndarray, bg: int) -> int:
    colors = sorted(set(map(int, grid.ravel())) - {bg})
    candidates = []
    for color in colors:
        comps = _components(grid == color, connectivity=8)
        if not comps or not all(len(comp) == 1 for comp in comps):
            continue
        coords = [tuple(map(int, comp[0])) for comp in comps]
        try:
            frames = _select_frames(coords)
        except AssertionError:
            continue
        if frames:
            candidates.append((len(frames), len(coords), color))
    assert candidates, "expected a marker color"
    candidates.sort()
    return int(candidates[-1][2])


def _outside_top_left(frame_bbox: tuple[int, int, int, int], corner: str, shape: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    h, w = shape
    if corner == "tl":
        return r0 - h, c0 - w
    if corner == "tr":
        return r0 - h, c1 + 1
    if corner == "bl":
        return r1 + 1, c0 - w
    if corner == "br":
        return r1 + 1, c1 + 1
    raise AssertionError(f"unknown corner {corner}")


def _attached_corner(frame_bbox: tuple[int, int, int, int], sprite_bbox: tuple[int, int, int, int]) -> str | None:
    r0, c0, r1, c1 = frame_bbox
    sr0, sc0, sr1, sc1 = sprite_bbox
    if (sr1, sc1) == (r0 - 1, c0 - 1):
        return "tl"
    if (sr1, sc0) == (r0 - 1, c1 + 1):
        return "tr"
    if (sr0, sc1) == (r1 + 1, c0 - 1):
        return "bl"
    if (sr0, sc0) == (r1 + 1, c1 + 1):
        return "br"
    return None


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    assert 0 <= top and 0 <= left
    assert top + h <= dst.shape[0] and left + w <= dst.shape[1]
    region = dst[top : top + h, left : left + w]
    mask = patch != bg
    assert np.all(region[mask] == bg)
    region[mask] = patch[mask]


def _sprite_components(grid: np.ndarray, bg: int, marker_color: int) -> list[dict]:
    mask = (grid != bg) & (grid != marker_color)
    out = []
    for coords in _components(mask, connectivity=4):
        r0, c0, r1, c1 = _bbox(coords)
        out.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": grid[r0 : r1 + 1, c0 : c1 + 1].copy(),
            }
        )
    return out


def _parse_dictionary(grid: np.ndarray) -> tuple[int, list[dict], dict[int, dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    marker_color = _find_marker_color(g, bg)
    marker_coords = [tuple(map(int, p)) for p in np.argwhere(g == marker_color)]
    frames = _select_frames(marker_coords)
    sprites = _sprite_components(g, bg, marker_color)

    by_side = {}
    for frame in frames:
        if frame["kind"] != "full":
            continue
        attached = []
        for sprite in sprites:
            corner = _attached_corner(frame["bbox"], sprite["bbox"])
            if corner is not None:
                attached.append((corner, sprite))
        assert len(attached) == 1, "every complete frame must have one attached sprite"
        corner, sprite = attached[0]
        side = int(frame["side"])
        assert side not in by_side, "complete-frame side lengths must be unique"
        by_side[side] = {
            "corner": corner,
            "patch": sprite["patch"],
        }
    return marker_color, frames, by_side


def main(grid: np.ndarray) -> np.ndarray:
    """Match each three-corner frame to the full frame of the same side length and stamp its sprite outside the missing corner.

    Input structure:
    - A dominant background contains many single-pixel corner markers of one color.
    - Some markers form complete four-corner square frames, each with one attached multicolor sprite placed just outside one corner.
    - The remaining marker sets form three-corner square frames with one missing corner; some of their side lengths have no matching complete frame.

    Rule:
    - Use each complete frame as a dictionary entry: its side length selects its attached sprite.
    - For every three-corner frame whose side length matches a complete frame, copy that sprite to the outside of the missing corner.
    - If the missing-corner quadrant differs from the source sprite's quadrant, reflect the sprite vertically and/or horizontally to match that corner.
    - Leave unmatched three-corner frames and all original pixels unchanged.

    Key insight:
    - The side length chooses which sprite to call, while the missing-corner quadrant chooses the orientation of that copied sprite.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    marker_color, frames, by_side = _parse_dictionary(g)
    out = g.copy()

    for frame in frames:
        if frame["kind"] != "missing":
            continue
        source = by_side.get(int(frame["side"]))
        if source is None:
            continue

        patch = source["patch"].copy()
        src_sign = CORNER_SIGNS[source["corner"]]
        dst_sign = CORNER_SIGNS[frame["missing_kind"]]
        if src_sign[0] != dst_sign[0]:
            patch = np.flipud(patch)
        if src_sign[1] != dst_sign[1]:
            patch = np.fliplr(patch)

        top, left = _outside_top_left(frame["bbox"], frame["missing_kind"], patch.shape)
        _paste(out, patch, top, left, bg)

    return out


def _random_connected_cells(rng: random.Random, h: int, w: int, size: int) -> list[tuple[int, int]]:
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    frontier = {start}
    while len(cells) < size and frontier:
        rr, cc = rng.choice(tuple(frontier))
        options = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                options.append((nr, nc))
        if not options:
            frontier.remove((rr, cc))
            continue
        chosen = rng.choice(options)
        cells.add(chosen)
        frontier.add(chosen)
    return sorted(cells)


def _make_sprite(rng: random.Random, bg: int, palette: list[int]) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        base, accent = rng.sample(palette, 2)
        patch = np.full((h, w), base, dtype=int)
        accent_count = rng.randint(2, min(5, h * w - 1))
        for r, c in _random_connected_cells(rng, h, w, accent_count):
            patch[r, c] = accent
        if np.array_equal(patch, np.fliplr(patch)):
            continue
        if np.array_equal(patch, np.flipud(patch)):
            continue
        if len(set(map(int, patch.ravel()))) < 2:
            continue
        return patch
    raise RuntimeError("failed to make asymmetric sprite")


def _object_bounds(spec: dict) -> tuple[int, int, int, int]:
    side = int(spec["side"])
    min_r = 0
    min_c = 0
    max_r = side
    max_c = side
    patch = spec.get("patch")
    if patch is None:
        return min_r, min_c, max_r, max_c
    h, w = patch.shape
    top, left = _outside_top_left((0, 0, side, side), spec["corner"], (h, w))
    min_r = min(min_r, top)
    min_c = min(min_c, left)
    max_r = max(max_r, top + h - 1)
    max_c = max(max_c, left + w - 1)
    return min_r, min_c, max_r, max_c


def _place_specs(rng: random.Random, h: int, w: int, specs: list[dict]) -> bool:
    occupied = np.zeros((h, w), dtype=bool)
    order = sorted(
        range(len(specs)),
        key=lambda i: (
            _object_bounds(specs[i])[2] - _object_bounds(specs[i])[0] + 1
        )
        * (_object_bounds(specs[i])[3] - _object_bounds(specs[i])[1] + 1),
        reverse=True,
    )

    for idx in order:
        spec = specs[idx]
        min_r, min_c, max_r, max_c = _object_bounds(spec)
        row_lo = -min_r
        row_hi = h - 1 - max_r
        col_lo = -min_c
        col_hi = w - 1 - max_c
        if row_lo > row_hi or col_lo > col_hi:
            return False

        candidates = []
        for top in range(row_lo, row_hi + 1):
            for left in range(col_lo, col_hi + 1):
                rr0 = top + min_r
                cc0 = left + min_c
                rr1 = top + max_r
                cc1 = left + max_c
                pr0 = max(0, rr0 - 1)
                pc0 = max(0, cc0 - 1)
                pr1 = min(h - 1, rr1 + 1)
                pc1 = min(w - 1, cc1 + 1)
                if occupied[pr0 : pr1 + 1, pc0 : pc1 + 1].any():
                    continue
                candidates.append((top, left))
        if not candidates:
            return False
        spec["top"], spec["left"] = rng.choice(candidates)
        rr0 = spec["top"] + min_r
        cc0 = spec["left"] + min_c
        rr1 = spec["top"] + max_r
        cc1 = spec["left"] + max_c
        occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return True


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], marker_color: int, missing_kind: str | None) -> None:
    r0, c0, r1, c1 = bbox
    corners = {
        "tl": (r0, c0),
        "tr": (r0, c1),
        "bl": (r1, c0),
        "br": (r1, c1),
    }
    for name, (r, c) in corners.items():
        if name == missing_kind:
            continue
        grid[r, c] = marker_color


def _build_grids(specs: list[dict], bg: int, marker_color: int, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, bg, dtype=int)
    out = np.full(shape, bg, dtype=int)

    for spec in specs:
        top = int(spec["top"])
        left = int(spec["left"])
        side = int(spec["side"])
        bbox = (top, left, top + side, left + side)
        if spec["type"] == "full":
            _draw_frame(inp, bbox, marker_color, None)
            _draw_frame(out, bbox, marker_color, None)
            patch = spec["patch"]
            pr, pc = _outside_top_left(bbox, spec["corner"], patch.shape)
            _paste(inp, patch, pr, pc, bg)
            _paste(out, patch, pr, pc, bg)
        else:
            _draw_frame(inp, bbox, marker_color, spec["corner"])
            _draw_frame(out, bbox, marker_color, spec["corner"])
            if spec.get("patch") is not None:
                patch = spec["patch"].copy()
                src_sign = CORNER_SIGNS[spec["source_corner"]]
                dst_sign = CORNER_SIGNS[spec["corner"]]
                if src_sign[0] != dst_sign[0]:
                    patch = np.flipud(patch)
                if src_sign[1] != dst_sign[1]:
                    patch = np.fliplr(patch)
                pr, pc = _outside_top_left(bbox, spec["corner"], patch.shape)
                _paste(out, patch, pr, pc, bg)
    return inp, out


def _semantic_check(inp: np.ndarray, out: np.ndarray) -> bool:
    produced = main(inp)
    if not np.array_equal(produced, out):
        return False
    marker_color, frames, by_side = _parse_dictionary(inp)
    if len(by_side) < 2:
        return False
    matched = [frame for frame in frames if frame["kind"] == "missing" and frame["side"] in by_side]
    unmatched = [frame for frame in frames if frame["kind"] == "missing" and frame["side"] not in by_side]
    if len(matched) < 2 or not unmatched:
        return False

    saw_same = False
    saw_flip = False
    for frame in matched:
        source = by_side[int(frame["side"])]
        if frame["missing_kind"] == source["corner"]:
            saw_same = True
        else:
            saw_flip = True
        h, w = source["patch"].shape
        top, left = _outside_top_left(frame["bbox"], frame["missing_kind"], (h, w))
        in_region = inp[top : top + h, left : left + w]
        out_region = out[top : top + h, left : left + w]
        if np.array_equal(in_region, out_region):
            return False
    for frame in unmatched:
        side = int(frame["side"])
        bbox = frame["bbox"]
        r0, c0, r1, c1 = bbox
        if not np.array_equal(inp[r0 : r1 + 1, c0 : c1 + 1], out[r0 : r1 + 1, c0 : c1 + 1]):
            return False
    return saw_same and saw_flip


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(23, 30)
    w = rng.randint(23, 30)
    bg = 0
    marker_color = rng.choice(PALETTE[1:]) if 0 in PALETTE else rng.choice(PALETTE)
    usable_colors = [c for c in PALETTE if c not in {bg, marker_color}]

    source_count = rng.randint(2, 3)
    all_sides = rng.sample(range(3, 8), source_count + 1)
    source_sides = all_sides[:source_count]
    distractor_side = all_sides[-1]

    source_infos = []
    for side in source_sides:
        source_infos.append(
            {
                "side": side,
                "corner": rng.choice(CORNER_NAMES),
                "patch": _make_sprite(rng, bg, usable_colors),
            }
        )

    specs = []
    for info in source_infos:
        specs.append(
            {
                "type": "full",
                "side": info["side"],
                "corner": info["corner"],
                "patch": info["patch"],
            }
        )

    anchor_info = source_infos[0]
    same_corner = anchor_info["corner"]
    different_corners = [corner for corner in CORNER_NAMES if corner != same_corner]
    specs.append(
        {
            "type": "missing",
            "side": anchor_info["side"],
            "corner": same_corner,
            "patch": anchor_info["patch"],
            "source_corner": anchor_info["corner"],
        }
    )
    specs.append(
        {
            "type": "missing",
            "side": anchor_info["side"],
            "corner": rng.choice(different_corners),
            "patch": anchor_info["patch"],
            "source_corner": anchor_info["corner"],
        }
    )

    for info in source_infos[1:]:
        count = rng.randint(1, 2)
        corners = rng.sample(list(CORNER_NAMES), count)
        for corner in corners:
            specs.append(
                {
                    "type": "missing",
                    "side": info["side"],
                    "corner": corner,
                    "patch": info["patch"],
                    "source_corner": info["corner"],
                }
            )

    specs.append(
        {
            "type": "missing",
            "side": distractor_side,
            "corner": rng.choice(CORNER_NAMES),
            "patch": None,
        }
    )
    if rng.random() < 0.45:
        extra_distractor = rng.choice([s for s in range(2, 9) if s not in source_sides and s != distractor_side])
        specs.append(
            {
                "type": "missing",
                "side": extra_distractor,
                "corner": rng.choice(CORNER_NAMES),
                "patch": None,
            }
        )

    if not _place_specs(rng, h, w, specs):
        return None

    inp, out = _build_grids(specs, bg, marker_color, (h, w))
    if np.array_equal(inp, out):
        return None
    if np.any(out == marker_color) and Counter(map(int, out.ravel())).most_common(1)[0][0] != bg:
        return None
    if not _semantic_check(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid missing-corner stamp puzzle")
