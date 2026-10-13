from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC = SourceFileLoader(
    "_arc_utils_fable_6_013_6",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC.Color
find_components = _ARC.find_components
get_bg = _ARC.get_bg


PALETTE = tuple(int(c) for c in Color.ALL_COLORS)
GEN_MAX_ATTEMPTS = 400


def _normalize_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    out = []
    for comp in comps:
        coords = tuple(sorted((int(r), int(c)) for r, c in np.asarray(comp["coords"], dtype=int)))
        out.append(
            {
                "coords": coords,
                "coord_key": frozenset(coords),
                "bbox": tuple(map(int, comp["bbox"])),
                "area": int(comp["area"]),
                "color": int(comp["color"]),
            }
        )
    out.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["bbox"][2], comp["bbox"][3], comp["color"]))
    return out


def _mirror_coords(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...], axis2: int) -> list[tuple[int, int]]:
    return [(int(r), int(axis2 - c)) for r, c in coords]


def _mirrored_key(coords: tuple[tuple[int, int], ...], axis2: int, width: int) -> frozenset[tuple[int, int]] | None:
    mirrored = []
    for r, c in coords:
        mc = int(axis2 - c)
        if not (0 <= mc < width):
            return None
        mirrored.append((int(r), mc))
    return frozenset(mirrored)


def _paired_indices(components: list[dict], axis2: int, width: int) -> set[int]:
    lookup: dict[tuple[int, frozenset[tuple[int, int]]], int] = {}
    for idx, comp in enumerate(components):
        lookup[(int(comp["color"]), comp["coord_key"])] = int(idx)

    paired: set[int] = set()
    for idx, comp in enumerate(components):
        mirrored = _mirrored_key(comp["coords"], axis2, width)
        if mirrored is None:
            continue
        partner = lookup.get((int(comp["color"]), mirrored))
        if partner is None:
            continue
        paired.add(int(idx))
        paired.add(int(partner))
    return paired


def _infer_axis2(grid: np.ndarray) -> tuple[int, list[dict], set[int]]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    assert height > 0 and width > 1

    bg = int(get_bg(g))
    components = _normalize_components(g, bg)
    assert components, "expected at least one foreground component"

    total_area = sum(int(comp["area"]) for comp in components)
    best: tuple[tuple[int, int, int], int, set[int]] | None = None
    for axis2 in range(0, 2 * width - 1):
        paired = _paired_indices(components, int(axis2), width)
        paired_area = sum(int(components[idx]["area"]) for idx in paired)
        if paired_area == 0:
            continue
        score = (int(paired_area), int(len(paired)), int(total_area - paired_area))
        if best is None or score > best[0]:
            best = (score, int(axis2), paired)

    assert best is not None, "failed to infer a mirror axis"
    _score, axis2, paired = best
    return int(axis2), components, paired


def _move_direction(coords: tuple[tuple[int, int], ...], axis2: int) -> int:
    cols2 = [2 * int(c) for _r, c in coords]
    if max(cols2) < int(axis2):
        return 1
    if min(cols2) > int(axis2):
        return -1
    if max(cols2) == int(axis2) and min(cols2) < int(axis2):
        return 1
    if min(cols2) == int(axis2) and max(cols2) > int(axis2):
        return -1
    raise AssertionError("unpaired mover should stay on one side of the axis")


def _crosses_axis(coords: list[tuple[int, int]], axis2: int, direction: int) -> bool:
    if int(direction) > 0:
        return any(2 * int(c) > int(axis2) for _r, c in coords)
    return any(2 * int(c) < int(axis2) for _r, c in coords)


def _transform_for_axis(grid: np.ndarray, axis2: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    bg = int(get_bg(g))
    components = _normalize_components(g, bg)
    paired = _paired_indices(components, int(axis2), width)

    assert paired, "expected a non-empty static scene"
    assert len(paired) < len(components), "expected at least one symmetry-breaking mover"

    out = np.full_like(g, bg)
    scene = np.full_like(g, bg)

    movers: list[dict] = []
    for idx, comp in enumerate(components):
        color = int(comp["color"])
        for r, c in comp["coords"]:
            if idx in paired:
                scene[int(r), int(c)] = color
                out[int(r), int(c)] = color
            else:
                movers.append(comp)

    for mover in movers:
        direction = _move_direction(mover["coords"], int(axis2))
        current = list(mover["coords"])
        final_color = int(mover["color"])

        while True:
            nxt = [(int(r), int(c + direction)) for r, c in current]
            if _crosses_axis(nxt, int(axis2), direction):
                break
            overlap = [int(scene[int(r), int(c)]) for r, c in nxt if int(scene[int(r), int(c)]) != int(bg)]
            if overlap:
                assert len(set(overlap)) == 1, "collision face should belong to one scene color"
                final_color = int(overlap[0])
                break
            current = nxt

        mirrored = _mirror_coords(current, int(axis2))
        for coords in (current, mirrored):
            for r, c in coords:
                assert 0 <= int(r) < height and 0 <= int(c) < width
                assert int(out[int(r), int(c)]) in {int(bg), int(final_color)}
                out[int(r), int(c)] = int(final_color)

    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Move the symmetry-breaking objects onto a hidden vertical mirror axis and restore symmetry.

    Input structure:
    - A dominant background contains several monochrome 4-connected objects.
    - Most objects form left-right mirror pairs around one hidden vertical axis; these paired objects are the static scene.
    - A smaller number of monochrome objects have no mirrored partner; these are the movers.

    Rule:
    - Infer the vertical axis from the objects that mirror exactly in shape and color.
    - Keep every paired scene object fixed.
    - For each unpaired mover, slide it horizontally toward the axis until one more step would cross the axis or overlap the static scene.
    - If that blocked step would hit the scene, recolor the whole mover to the touched scene color; otherwise keep its original color.
    - Remove the mover from its starting position and draw both its stopped position and its mirror image on the opposite side.

    Key insight: The objects that move are selected only by symmetry breaking: paired components define both the hidden axis and the stationary scene, while unpaired components translate, optionally recolor on collision, and then get mirrored to complete the scene.
    """
    g = np.asarray(input_grid, dtype=int)
    axis2, _components, _paired = _infer_axis2(g)
    return _transform_for_axis(g, int(axis2))


def _shape_candidates(h: int, w: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    masks.append(np.ones((h, w), dtype=bool))

    if h >= 2 and w >= 2:
        frame = np.zeros((h, w), dtype=bool)
        frame[0, :] = True
        frame[-1, :] = True
        frame[:, 0] = True
        frame[:, -1] = True
        masks.append(frame)

        tl = np.zeros((h, w), dtype=bool)
        tl[0, :] = True
        tl[:, 0] = True
        masks.append(tl)

        tr = np.zeros((h, w), dtype=bool)
        tr[0, :] = True
        tr[:, -1] = True
        masks.append(tr)

        bl = np.zeros((h, w), dtype=bool)
        bl[-1, :] = True
        bl[:, 0] = True
        masks.append(bl)

        br = np.zeros((h, w), dtype=bool)
        br[-1, :] = True
        br[:, -1] = True
        masks.append(br)

    if h >= 2 and w >= 3:
        mid = w // 2
        t_up = np.zeros((h, w), dtype=bool)
        t_up[0, :] = True
        t_up[:, mid] = True
        masks.append(t_up)

        t_down = np.zeros((h, w), dtype=bool)
        t_down[-1, :] = True
        t_down[:, mid] = True
        masks.append(t_down)

    if h >= 3 and w >= 2:
        mid = h // 2
        t_left = np.zeros((h, w), dtype=bool)
        t_left[:, 0] = True
        t_left[mid, :] = True
        masks.append(t_left)

        t_right = np.zeros((h, w), dtype=bool)
        t_right[:, -1] = True
        t_right[mid, :] = True
        masks.append(t_right)

    if h >= 3 and w >= 3:
        plus = np.zeros((h, w), dtype=bool)
        plus[h // 2, :] = True
        plus[:, w // 2] = True
        masks.append(plus)

    dedup: dict[bytes, np.ndarray] = {}
    for mask in masks:
        dedup.setdefault(mask.astype(np.uint8).tobytes(), mask)
    return list(dedup.values())


def _random_shape(max_h: int = 4, max_w: int = 4) -> np.ndarray:
    h = random.randint(1, int(max_h))
    w = random.randint(1, int(max_w))
    return random.choice(_shape_candidates(h, w)).copy()


def _place_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    rows, cols = np.where(mask)
    for r, c in zip(rows, cols):
        rr = int(top + int(r))
        cc = int(left + int(c))
        grid[rr, cc] = int(color)
        coords.append((rr, cc))
    return coords


def _lane_contact_ok(scene_mask: np.ndarray, mover_mask: np.ndarray, kind: str) -> bool:
    if kind == "collision_left":
        return bool(np.any(scene_mask[:, 0] & mover_mask[:, -1]))
    if kind == "collision_right":
        return bool(np.any(scene_mask[:, -1] & mover_mask[:, 0]))
    return True


def _lane_placements(width: int, axis2: int, scene_mask: np.ndarray, mover_mask: np.ndarray | None, kind: str) -> list[dict]:
    m = int(axis2 // 2)
    scene_h, scene_w = map(int, scene_mask.shape)
    left_min = max(1, int(axis2 - (width - 1)))
    left_max = int(m - scene_w + 1)
    placements: list[dict] = []
    if left_min > left_max:
        return placements

    for scene_left in range(left_min, left_max + 1):
        right_scene_left = int(axis2 - (scene_left + scene_w - 1))
        if right_scene_left < 0 or right_scene_left + scene_w > width:
            continue

        base = {"scene_left": int(scene_left)}
        if kind == "plain":
            placements.append(base)
            continue

        assert mover_mask is not None
        mover_h, mover_w = map(int, mover_mask.shape)

        if kind == "collision_left":
            for gap in range(1, 5):
                mover_left = int(scene_left - mover_w - gap)
                if mover_left < 1:
                    continue
                placements.append({"scene_left": int(scene_left), "mover_left": mover_left})

        elif kind == "collision_right":
            for gap in range(1, 5):
                mover_left = int(right_scene_left + scene_w + gap)
                if mover_left + mover_w > width - 1:
                    continue
                placements.append({"scene_left": int(scene_left), "mover_left": mover_left})

        elif kind == "free_left":
            final_left = int(m - mover_w + 1)
            lo = int(scene_left + scene_w + 1)
            hi = int(final_left)
            for mover_left in range(lo, hi + 1):
                placements.append({"scene_left": int(scene_left), "mover_left": int(mover_left)})

        elif kind == "free_right":
            final_left = int(m + 1)
            lo = int(final_left)
            hi = int(right_scene_left - mover_w - 1)
            for mover_left in range(lo, hi + 1):
                placements.append({"scene_left": int(scene_left), "mover_left": int(mover_left)})

    return placements


def _random_lane(bg: int, width: int, axis2: int, kind: str) -> dict | None:
    mover_needed = kind != "plain"
    for _ in range(120):
        if kind.startswith("collision"):
            shared_h = random.randint(1, 4)
            scene_mask = random.choice(_shape_candidates(shared_h, random.randint(1, 4))).copy()
            mover_mask = random.choice(_shape_candidates(shared_h, random.randint(1, 4))).copy()
            if not _lane_contact_ok(scene_mask, mover_mask, kind):
                continue
        else:
            scene_mask = _random_shape()
            mover_mask = _random_shape() if mover_needed else None

        placements = _lane_placements(width, axis2, scene_mask, mover_mask, kind)
        if not placements:
            continue

        scene_color = random.choice([c for c in PALETTE if int(c) != int(bg)])
        mover_color = None
        if mover_needed:
            mover_choices = [c for c in PALETTE if int(c) != int(bg)]
            if kind.startswith("collision"):
                mover_choices = [c for c in mover_choices if int(c) != int(scene_color)]
            mover_color = random.choice(mover_choices)

        lane_h = max(int(scene_mask.shape[0]), int(mover_mask.shape[0]) if mover_mask is not None else 0)
        placement = random.choice(placements)
        return {
            "kind": kind,
            "height": int(lane_h),
            "scene_mask": scene_mask,
            "scene_color": int(scene_color),
            "scene_left": int(placement["scene_left"]),
            "mover_mask": mover_mask,
            "mover_color": None if mover_color is None else int(mover_color),
            "mover_left": None if "mover_left" not in placement else int(placement["mover_left"]),
        }
    return None


def _paint_lane(grid: np.ndarray, lane: dict, top: int, axis2: int) -> None:
    scene_mask = lane["scene_mask"]
    scene_left = int(lane["scene_left"])
    scene_color = int(lane["scene_color"])
    left_coords = _place_mask(grid, scene_mask, int(top), int(scene_left), scene_color)
    right_coords = _mirror_coords(left_coords, int(axis2))
    for r, c in right_coords:
        grid[int(r), int(c)] = scene_color

    mover_mask = lane["mover_mask"]
    if mover_mask is not None:
        _place_mask(grid, mover_mask, int(top), int(lane["mover_left"]), int(lane["mover_color"]))


def _is_symmetric(grid: np.ndarray, axis2: int) -> bool:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    for r in range(height):
        for c in range(width):
            mc = int(axis2 - c)
            if 0 <= mc < width and int(g[r, c]) != int(g[r, mc]):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(16, 28)
    left_axis_col = random.randint(5, width - 7)
    axis2 = int(2 * left_axis_col + 1)

    bg = random.choice(PALETTE)
    lane_count = random.randint(3, 5)
    mover_count = random.randint(1, min(3, lane_count))
    mover_rows = set(random.sample(range(lane_count), mover_count))

    lane_kinds: list[str] = []
    mover_kinds = ["collision_left", "collision_right", "free_left", "free_right"]
    for lane_idx in range(lane_count):
        if lane_idx in mover_rows:
            lane_kinds.append(random.choices(mover_kinds, weights=[3, 3, 2, 2], k=1)[0])
        else:
            lane_kinds.append("plain")

    lanes: list[dict] = []
    for kind in lane_kinds:
        lane = _random_lane(int(bg), width, int(axis2), kind)
        if lane is None:
            return None
        lanes.append(lane)

    height = 1 + sum(int(lane["height"]) + 1 for lane in lanes)
    if height > 30:
        return None

    grid = np.full((height, width), int(bg), dtype=int)
    top = 1
    for lane in lanes:
        _paint_lane(grid, lane, int(top), int(axis2))
        top += int(lane["height"]) + 1

    try:
        inferred_axis2, components, paired = _infer_axis2(grid)
        if int(inferred_axis2) != int(axis2):
            return None
        if len(paired) >= len(components):
            return None
        expected = _transform_for_axis(grid, int(axis2))
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    if not _is_symmetric(actual, int(axis2)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to construct a valid symmetry-restoration instance")
