from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 8
FRAME = 5
PIECE_COLORS = [0, 1, 2, 3, 4, 6, 7, 9]
GEN_RETRIES = 300
GRID_MIN = 18
GRID_MAX = 30
PAIR_COUNT_CHOICES = (2, 3, 4)
PLACEMENT_PADDING = 1
PLACEMENT_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _components_by_color(
    grid: np.ndarray, bg: int, blocked: np.ndarray | None = None
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    blocked_mask = np.zeros((h, w), dtype=bool) if blocked is None else np.asarray(blocked, dtype=bool)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or blocked_mask[r, c]:
                continue
            color = int(g[r, c])
            if color == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or blocked_mask[nr, nc]:
                        continue
                    if int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append({"color": color, "coords": pts})
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows, cols = zip(*coords)
    return min(rows), min(cols), max(rows), max(cols)


def _coords_to_mask(coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components_from_mask(mask)
    return len(comps) == 1


def _full_rect_border(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts = {(r0, c) for c in range(c0, c1 + 1)}
    pts.update({(r1, c) for c in range(c0, c1 + 1)})
    pts.update({(r, c0) for r in range(r0, r1 + 1)})
    pts.update({(r, c1) for r in range(r0, r1 + 1)})
    return pts


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for comp in _components_by_color(grid, bg):
        coords = comp["coords"]
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue
        border = _full_rect_border(r0, c0, r1, c1)
        coord_set = set(coords)
        if not coord_set.issubset(border):
            continue
        if len(coord_set) != len(border) - 1:
            continue
        missing = list(border - coord_set)
        if len(missing) != 1:
            continue
        mr, mc = missing[0]
        if mr != r0:
            continue
        marker_color = int(grid[mr, mc])
        if marker_color in (bg, comp["color"]):
            continue
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size and not np.all(interior == bg):
            continue
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": (r1 - r0 - 1, c1 - c0 - 1),
                "frame_color": int(comp["color"]),
                "marker_pos": (mr, mc),
                "marker_color": marker_color,
                "border": border,
            }
        )
    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def _find_fragments(grid: np.ndarray, bg: int, frames: list[dict]) -> list[dict]:
    blocked = np.zeros_like(grid, dtype=bool)
    for frame in frames:
        for r, c in frame["border"]:
            blocked[r, c] = True
        mr, mc = frame["marker_pos"]
        blocked[mr, mc] = True

    fragments: list[dict] = []
    for comp in _components_by_color(grid, bg, blocked=blocked):
        coords = comp["coords"]
        bb = _bbox(coords)
        mask = _coords_to_mask(coords, bb)
        fragments.append(
            {
                "color": int(comp["color"]),
                "coords": coords,
                "bbox": bb,
                "shape": mask.shape,
                "mask": mask,
                "area": int(mask.sum()),
            }
        )
    fragments.sort(key=lambda frag: (frag["bbox"], frag["color"]))
    return fragments


def _enumerate_arrangements(frag_a: dict, frag_b: dict, rect_shape: tuple[int, int]) -> list[dict]:
    rh, rw = rect_shape
    if frag_a["area"] + frag_b["area"] != rh * rw:
        return []
    ah, aw = frag_a["shape"]
    bh, bw = frag_b["shape"]
    if ah > rh or aw > rw or bh > rh or bw > rw:
        return []

    arrangements: list[dict] = []
    for ar in range(rh - ah + 1):
        for ac in range(rw - aw + 1):
            board_a = np.zeros((rh, rw), dtype=bool)
            board_a[ar : ar + ah, ac : ac + aw] = frag_a["mask"]
            for br in range(rh - bh + 1):
                for bc in range(rw - bw + 1):
                    board_b = np.zeros((rh, rw), dtype=bool)
                    board_b[br : br + bh, bc : bc + bw] = frag_b["mask"]
                    if np.any(board_a & board_b):
                        continue
                    if not np.all(board_a | board_b):
                        continue
                    arrangements.append(
                        {
                            "shape": rect_shape,
                            "a_off": (ar, ac),
                            "b_off": (br, bc),
                        }
                    )
    return arrangements


def _candidate_arrangements(fragments: list[dict], rect_shapes: list[tuple[int, int]]) -> list[dict]:
    arrangements: list[dict] = []
    unique_shapes = sorted(set(rect_shapes))
    for i in range(len(fragments)):
        for j in range(i + 1, len(fragments)):
            if fragments[i]["color"] == fragments[j]["color"]:
                continue
            for shape in unique_shapes:
                for placement in _enumerate_arrangements(fragments[i], fragments[j], shape):
                    arrangements.append({"a": i, "b": j, **placement})
    return arrangements


def _assign_frames(frames: list[dict], fragments: list[dict], arrangements: list[dict]) -> dict[int, int]:
    assert len(fragments) == 2 * len(frames), "fragment count must equal twice the frame count"
    candidates: list[list[int]] = []
    for frame in frames:
        valid: list[int] = []
        for idx, arrangement in enumerate(arrangements):
            if arrangement["shape"] != frame["shape"]:
                continue
            colors = {fragments[arrangement["a"]]["color"], fragments[arrangement["b"]]["color"]}
            if frame["marker_color"] in colors:
                valid.append(idx)
        assert valid, "each frame must have at least one compatible fragment pair"
        candidates.append(valid)

    order = sorted(range(len(frames)), key=lambda i: len(candidates[i]))
    solutions: list[dict[int, int]] = []
    used_pairs: set[int] = set()
    used_fragments: set[int] = set()
    current: dict[int, int] = {}

    def dfs(depth: int) -> None:
        if len(solutions) > 1:
            return
        if depth == len(order):
            if len(used_fragments) == len(fragments):
                solutions.append(current.copy())
            return
        frame_idx = order[depth]
        for arrangement_idx in candidates[frame_idx]:
            arrangement = arrangements[arrangement_idx]
            a_idx = arrangement["a"]
            b_idx = arrangement["b"]
            if arrangement_idx in used_pairs or a_idx in used_fragments or b_idx in used_fragments:
                continue
            used_pairs.add(arrangement_idx)
            used_fragments.add(a_idx)
            used_fragments.add(b_idx)
            current[frame_idx] = arrangement_idx
            dfs(depth + 1)
            current.pop(frame_idx)
            used_fragments.remove(a_idx)
            used_fragments.remove(b_idx)
            used_pairs.remove(arrangement_idx)

    dfs(0)
    assert len(solutions) == 1, "frame-to-pair assignment must be unique"
    return solutions[0]


def _paint_mask(out: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    view = out[top : top + h, left : left + w]
    view[mask] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble complementary fragment pairs into the matching marked hollow frames."""
    g = np.asarray(grid, dtype=int)
    bg = int(np.bincount(g.reshape(-1), minlength=10).argmax())
    frames = _find_frames(g, bg)
    assert frames, "expected at least one hollow frame"
    fragments = _find_fragments(g, bg, frames)
    assert fragments, "expected at least one fragment"
    arrangements = _candidate_arrangements(fragments, [frame["shape"] for frame in frames])
    assignment = _assign_frames(frames, fragments, arrangements)

    out = g.copy()
    for fragment in fragments:
        for r, c in fragment["coords"]:
            out[r, c] = bg

    for frame_idx, arrangement_idx in assignment.items():
        frame = frames[frame_idx]
        arrangement = arrangements[arrangement_idx]
        frag_a = fragments[arrangement["a"]]
        frag_b = fragments[arrangement["b"]]
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = bg
        if frag_a["color"] == frame["marker_color"]:
            front = (frag_a, arrangement["a_off"])
            back = (frag_b, arrangement["b_off"])
        else:
            front = (frag_b, arrangement["b_off"])
            back = (frag_a, arrangement["a_off"])
        _paint_mask(out, r0 + 1 + back[1][0], c0 + 1 + back[1][1], back[0]["mask"], back[0]["color"])
        _paint_mask(out, r0 + 1 + front[1][0], c0 + 1 + front[1][1], front[0]["mask"], front[0]["color"])
    return out


def _mask_from_bits(bits: int, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for idx in range(h * w):
        if bits & (1 << idx):
            mask[idx // w, idx % w] = True
    return mask


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    coords = [tuple(map(int, pt)) for pt in np.argwhere(mask)]
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0)


def _build_pair_library() -> list[dict]:
    library: list[dict] = []
    for h in range(2, 7):
        for w in range(2, 7):
            area = h * w
            if area > 12:
                continue
            full_bits = (1 << area) - 1
            for bits in range(1, full_bits):
                count = bits.bit_count()
                other = area - count
                if count > 6 or other > 6 or count < 2 or other < 2:
                    continue
                comp_bits = full_bits ^ bits
                if bits > comp_bits:
                    continue
                rect_mask_a = _mask_from_bits(bits, h, w)
                rect_mask_b = ~rect_mask_a
                if not (_is_connected(rect_mask_a) and _is_connected(rect_mask_b)):
                    continue
                mask_a, off_a = _crop_mask(rect_mask_a)
                mask_b, off_b = _crop_mask(rect_mask_b)
                library.append(
                    {
                        "shape": (h, w),
                        "bits": bits,
                        "mask_a": mask_a,
                        "mask_b": mask_b,
                        "off_a": off_a,
                        "off_b": off_b,
                    }
                )
    assert library, "pair library must not be empty"
    return library


PAIR_LIBRARY = _build_pair_library()
PAIRS_BY_SHAPE: dict[tuple[int, int], list[dict]] = {}
for item in PAIR_LIBRARY:
    PAIRS_BY_SHAPE.setdefault(item["shape"], []).append(item)


def _choose_pair_specs(pair_count: int) -> list[dict]:
    reuse_shape = pair_count >= 2 and random.random() < 0.5
    chosen: list[dict] = []
    used_keys: set[tuple[tuple[int, int], int]] = set()
    if reuse_shape:
        repeat_shapes = [shape for shape, items in PAIRS_BY_SHAPE.items() if len(items) >= 2]
        shape = random.choice(repeat_shapes)
        max_repeat = min(pair_count, len(PAIRS_BY_SHAPE[shape]))
        repeat_n = random.randint(2, max_repeat)
        picks = random.sample(PAIRS_BY_SHAPE[shape], repeat_n)
        for pick in picks:
            chosen.append(pick)
            used_keys.add((pick["shape"], pick["bits"]))
    while len(chosen) < pair_count:
        pick = random.choice(PAIR_LIBRARY)
        key = (pick["shape"], pick["bits"])
        if key in used_keys:
            continue
        if not reuse_shape:
            used_shapes = {item["shape"] for item in chosen}
            if pick["shape"] in used_shapes and random.random() < 0.75:
                continue
        chosen.append(pick)
        used_keys.add(key)
    random.shuffle(chosen)
    return chosen


def _fragment_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = color
    return patch


def _frame_patch(inner_shape: tuple[int, int], marker_color: int) -> tuple[np.ndarray, int]:
    h, w = inner_shape
    patch = np.full((h + 2, w + 2), BG, dtype=int)
    patch[0, :] = FRAME
    patch[-1, :] = FRAME
    patch[:, 0] = FRAME
    patch[:, -1] = FRAME
    marker_c = random.randint(0, w + 1)
    patch[0, marker_c] = marker_color
    return patch, marker_c


def _can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int, padding: int) -> bool:
    oh, ow = occupied.shape
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(oh, top + h + padding)
    c1 = min(ow, left + w + padding)
    return not occupied[r0:r1, c0:c1].any()


def _place_patch(canvas: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    h, w = patch.shape
    oh, ow = canvas.shape
    if h + 2 > oh or w + 2 > ow:
        return None
    positions = [(r, c) for r in range(1, oh - h) for c in range(1, ow - w)]
    if not positions:
        return None
    random.shuffle(positions)
    for top, left in positions[:PLACEMENT_ATTEMPTS]:
        if not _can_place(occupied, top, left, h, w, PLACEMENT_PADDING):
            continue
        view = canvas[top : top + h, left : left + w]
        mask = patch != BG
        view[mask] = patch[mask]
        occupied[top : top + h, left : left + w] = True
        return top, left
    return None


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    min_side = max(GRID_MIN, 14 + 2 * pair_count)
    h = random.randint(min_side, GRID_MAX)
    w = random.randint(min_side, GRID_MAX)
    pair_specs = _choose_pair_specs(pair_count)
    colors = random.sample(PIECE_COLORS, 2 * pair_count)

    inp = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    pair_data: list[dict] = []
    for idx, spec in enumerate(pair_specs):
        color_a = colors[2 * idx]
        color_b = colors[2 * idx + 1]
        if random.random() < 0.5:
            color_a, color_b = color_b, color_a
        marker_color = random.choice((color_a, color_b))
        frame_patch, _ = _frame_patch(spec["shape"], marker_color)
        frame_pos = _place_patch(inp, occupied, frame_patch)
        if frame_pos is None:
            return None
        fr, fc = frame_pos
        frame_view = expected[fr : fr + frame_patch.shape[0], fc : fc + frame_patch.shape[1]]
        frame_view[frame_patch != BG] = frame_patch[frame_patch != BG]
        pair_data.append(
            {
                "shape": spec["shape"],
                "mask_a": spec["mask_a"],
                "mask_b": spec["mask_b"],
                "off_a": spec["off_a"],
                "off_b": spec["off_b"],
                "color_a": color_a,
                "color_b": color_b,
                "marker_color": marker_color,
                "frame_pos": frame_pos,
            }
        )

    fragments_to_place: list[tuple[np.ndarray, int, np.ndarray, tuple[int, int]]] = []
    for pair in pair_data:
        fragments_to_place.append((pair["mask_a"], pair["color_a"], pair["mask_b"], pair["frame_pos"]))
        fragments_to_place.append((pair["mask_b"], pair["color_b"], pair["mask_a"], pair["frame_pos"]))

    random.shuffle(fragments_to_place)
    for mask, color, _other_mask, _frame_pos in fragments_to_place:
        frag_patch = _fragment_patch(mask, color)
        if _place_patch(inp, occupied, frag_patch) is None:
            return None

    for pair in pair_data:
        fr, fc = pair["frame_pos"]
        inner_top = fr + 1
        inner_left = fc + 1
        expected[inner_top : inner_top + pair["shape"][0], inner_left : inner_left + pair["shape"][1]] = BG
        _paint_mask(
            expected,
            inner_top + pair["off_a"][0],
            inner_left + pair["off_a"][1],
            pair["mask_a"],
            pair["color_a"],
        )
        _paint_mask(
            expected,
            inner_top + pair["off_b"][0],
            inner_left + pair["off_b"][1],
            pair["mask_b"],
            pair["color_b"],
        )

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid complementary-frame ARC input")
