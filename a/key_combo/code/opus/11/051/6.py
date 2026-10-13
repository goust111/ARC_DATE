from __future__ import annotations

import itertools
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_ROWS = (18, 27)
GRID_COLS = (18, 27)
DOMINO_COUNTS = (2, 3)
MAX_SINGLES = 4
MAX_GENERATION_ATTEMPTS = 400

MATCHED_CAVITIES = {
    "H": (3, 4),  # outlined 1x2 domino
    "V": (4, 3),  # outlined 2x1 domino
}
DISTRACTOR_CAVITIES = ((4, 4), (3, 5), (5, 3))


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = np.asarray(grid == int(color), dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_hw(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _box_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b

    def _gap1(x0: int, x1: int, y0: int, y1: int) -> int:
        if x1 < y0:
            return y0 - x1 - 1
        if y1 < x0:
            return x0 - y1 - 1
        return 0

    return _gap1(ar0, ar1, br0, br1) + _gap1(ac0, ac1, bc0, bc1)


def _domino_kind(coords: np.ndarray) -> str:
    assert coords.shape[0] == 2
    box = _bbox(coords)
    h, w = _bbox_hw(box)
    assert (h, w) in {(1, 2), (2, 1)}
    return "H" if w == 2 else "V"


def _outlined_patch_from_domino(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    local = coords - np.array([r0 - 1, c0 - 1], dtype=int)
    patch = np.zeros((r1 - r0 + 3, c1 - c0 + 3), dtype=int)
    patch[0, :] = GREEN
    patch[-1, :] = GREEN
    patch[:, 0] = GREEN
    patch[:, -1] = GREEN
    patch[local[:, 0], local[:, 1]] = RED
    return patch


def _outlined_patch_for_kind(kind: str) -> np.ndarray:
    if kind == "H":
        coords = np.array([[0, 0], [0, 1]], dtype=int)
    else:
        coords = np.array([[0, 0], [1, 0]], dtype=int)
    return _outlined_patch_from_domino(coords)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = BLUE
    grid[r1, c0 : c1 + 1] = BLUE
    grid[r0 : r1 + 1, c0] = BLUE
    grid[r0 : r1 + 1, c1] = BLUE


def _frame_info_from_component(comp: np.ndarray) -> dict:
    box = _bbox(comp)
    r0, c0, r1, c1 = box
    cavity_box = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
    cavity_hw = _bbox_hw(cavity_box)
    return {
        "frame_box": box,
        "cavity_box": cavity_box,
        "cavity_hw": cavity_hw,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Outline every red domino, then dock the outlined figure into the blue frame cavity with the same post-outline bounding-box size.

    Input structure:
    - The background is black.
    - Blue pixels form one-pixel-thick rectangular frames whose interiors are black rectangular cavities.
    - Red 4-connected components are either isolated single pixels or 2-cell horizontal/vertical dominoes.

    Rule:
    - Keep every blue frame unchanged.
    - Keep each single red pixel in place.
    - For each red domino, expand its bbox by one cell on every side and color the perimeter green, producing an outlined figure containing the original two red cells.
    - Match each outlined figure to a blue frame cavity whose height and width exactly equal the outlined figure's bbox height and width.
    - Erase the domino from its original location and translate the whole outlined figure into the matching cavity so the two bboxes coincide.

    Key insight:
    - The blue cavities match the dominoes only after outlining: raw 1x2 or 2x1 dominoes fit nothing, while outlined horizontal and vertical dominoes become distinct 3x4 and 4x3 figures.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()

    frames = [_frame_info_from_component(comp) for comp in _components_of_color(g, BLUE)]
    frame_counts = Counter(frame["cavity_hw"] for frame in frames)

    dominoes: list[np.ndarray] = []
    singles: list[np.ndarray] = []
    for comp in _components_of_color(g, RED):
        if comp.shape[0] == 1:
            singles.append(comp)
            continue
        assert comp.shape[0] == 2
        _domino_kind(comp)
        dominoes.append(comp)

    placed_counts: Counter[tuple[int, int]] = Counter()
    for comp in dominoes:
        patch = _outlined_patch_from_domino(comp)
        hw = tuple(map(int, patch.shape))
        assert frame_counts[hw] > 0, "every domino must have at least one size-compatible cavity"
        placed_counts[hw] += 1

    for hw, count in placed_counts.items():
        assert count <= frame_counts[hw], "not enough matching cavities for the outlined dominoes"

    for comp in dominoes:
        out[comp[:, 0], comp[:, 1]] = BLACK

    frames_by_hw: dict[tuple[int, int], list[dict]] = {}
    for frame in frames:
        frames_by_hw.setdefault(frame["cavity_hw"], []).append(frame)

    for comp in dominoes:
        patch = _outlined_patch_from_domino(comp)
        cavity = frames_by_hw[tuple(map(int, patch.shape))].pop()
        cr0, cc0, _cr1, _cc1 = cavity["cavity_box"]
        _paste(out, patch, cr0, cc0)

    return out


def _naive_without_outline(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    frames = [_frame_info_from_component(comp) for comp in _components_of_color(out, BLUE)]
    frames_by_hw: dict[tuple[int, int], list[dict]] = {}
    for frame in frames:
        frames_by_hw.setdefault(frame["cavity_hw"], []).append(frame)

    dominoes = [comp for comp in _components_of_color(out, RED) if comp.shape[0] == 2]
    for comp in dominoes:
        box = _bbox(comp)
        hw = _bbox_hw(box)
        if hw not in frames_by_hw or not frames_by_hw[hw]:
            continue
        cavity = frames_by_hw[hw].pop()
        out[comp[:, 0], comp[:, 1]] = BLACK
        cr0, cc0, _cr1, _cc1 = cavity["cavity_box"]
        local = comp - np.array([box[0], box[1]], dtype=int)
        for rr, cc in local:
            out[cr0 + rr, cc0 + cc] = RED
    return out


def _candidate_frame_boxes(h: int, w: int, outer_h: int, outer_w: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    edge_offsets = (0, 1, 2)

    for r0 in edge_offsets:
        if r0 + outer_h <= h:
            for c0 in range(0, w - outer_w + 1):
                boxes.append((r0, c0, r0 + outer_h - 1, c0 + outer_w - 1))
    for r0 in range(max(0, h - outer_h - 2), h - outer_h + 1):
        if 0 <= r0 <= h - outer_h:
            for c0 in range(0, w - outer_w + 1):
                boxes.append((r0, c0, r0 + outer_h - 1, c0 + outer_w - 1))
    for c0 in edge_offsets:
        if c0 + outer_w <= w:
            for r0 in range(0, h - outer_h + 1):
                boxes.append((r0, c0, r0 + outer_h - 1, c0 + outer_w - 1))
    for c0 in range(max(0, w - outer_w - 2), w - outer_w + 1):
        if 0 <= c0 <= w - outer_w:
            for r0 in range(0, h - outer_h + 1):
                boxes.append((r0, c0, r0 + outer_h - 1, c0 + outer_w - 1))

    uniq = list(dict.fromkeys(boxes))
    return uniq


def _sample_frame_layout(
    rng: np.random.Generator,
    h: int,
    w: int,
    cavity_list: list[tuple[int, int]],
) -> list[dict] | None:
    specs = []
    for cavity_h, cavity_w in cavity_list:
        outer_h = cavity_h + 2
        outer_w = cavity_w + 2
        boxes = _candidate_frame_boxes(h, w, outer_h, outer_w)
        rng.shuffle(boxes)
        specs.append(
            {
                "cavity_hw": (cavity_h, cavity_w),
                "outer_hw": (outer_h, outer_w),
                "boxes": boxes,
            }
        )

    order = np.argsort([-len(spec["boxes"]) for spec in specs])
    chosen: list[dict] = []
    used_boxes: list[tuple[int, int, int, int]] = []

    def rec(i: int) -> bool:
        if i == len(order):
            return True
        spec = specs[int(order[i])]
        for box in spec["boxes"]:
            if any(_box_gap(box, other) < 2 for other in used_boxes):
                continue
            used_boxes.append(box)
            chosen.append(
                {
                    "frame_box": box,
                    "cavity_box": (box[0] + 1, box[1] + 1, box[2] - 1, box[3] - 1),
                    "cavity_hw": spec["cavity_hw"],
                }
            )
            if rec(i + 1):
                return True
            chosen.pop()
            used_boxes.pop()
        return False

    if not rec(0):
        return None
    rng.shuffle(chosen)
    return chosen


def _mark_forbidden(mask: np.ndarray, box: tuple[int, int, int, int], pad: int) -> None:
    r0, c0, r1, c1 = box
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _domino_cells(r: int, c: int, kind: str) -> np.ndarray:
    if kind == "H":
        return np.array([[r, c], [r, c + 1]], dtype=int)
    return np.array([[r, c], [r + 1, c]], dtype=int)


def _enumerate_domino_positions(
    forbidden: np.ndarray,
    frames: list[dict],
    kind: str,
) -> list[np.ndarray]:
    h, w = forbidden.shape
    cells = []
    max_r = h if kind == "H" else h - 1
    max_c = w - 1 if kind == "H" else w
    for r in range(max_r):
        for c in range(max_c):
            comp = _domino_cells(r, c, kind)
            box = _bbox(comp)
            expanded = (box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1)
            if expanded[0] < 0 or expanded[1] < 0 or expanded[2] >= h or expanded[3] >= w:
                continue
            if forbidden[expanded[0] : expanded[2] + 1, expanded[1] : expanded[3] + 1].any():
                continue
            if any(_box_gap(expanded, frame["frame_box"]) < 4 for frame in frames):
                continue
            cells.append(comp)
    return cells


def _enumerate_single_positions(forbidden: np.ndarray) -> list[tuple[int, int]]:
    h, w = forbidden.shape
    out = []
    for r in range(h):
        for c in range(w):
            if forbidden[r, c]:
                continue
            out.append((r, c))
    return out


def _build_expected_output(
    h: int,
    w: int,
    frames: list[dict],
    kinds: list[str],
    singles: list[tuple[int, int]],
) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for frame in frames:
        _draw_frame(out, frame["frame_box"])
    for frame, kind in zip(frames, kinds):
        if frame["cavity_hw"] not in MATCHED_CAVITIES.values():
            continue
        cr0, cc0, _cr1, _cc1 = frame["cavity_box"]
        _paste(out, _outlined_patch_for_kind(kind), cr0, cc0)
    for r, c in singles:
        out[r, c] = RED
    return out


def _build_input_grid(
    h: int,
    w: int,
    frames: list[dict],
    dominoes: list[np.ndarray],
    singles: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for frame in frames:
        _draw_frame(grid, frame["frame_box"])
    for comp in dominoes:
        grid[comp[:, 0], comp[:, 1]] = RED
    for r, c in singles:
        grid[r, c] = RED
    return grid


def _validate_blue_frames(grid: np.ndarray) -> list[dict]:
    frames = [_frame_info_from_component(comp) for comp in _components_of_color(grid, BLUE)]
    assert 2 <= len(frames) <= 4
    for frame in frames:
        r0, c0, r1, c1 = frame["frame_box"]
        assert r1 - r0 >= 4 and c1 - c0 >= 4
        assert np.all(grid[r0, c0 : c1 + 1] == BLUE)
        assert np.all(grid[r1, c0 : c1 + 1] == BLUE)
        assert np.all(grid[r0 : r1 + 1, c0] == BLUE)
        assert np.all(grid[r0 : r1 + 1, c1] == BLUE)
        cr0, cc0, cr1, cc1 = frame["cavity_box"]
        assert np.all(grid[cr0 : cr1 + 1, cc0 : cc1 + 1] == BLACK)
    return frames


def _validate_red_components(grid: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    singles = []
    dominoes = []
    for comp in _components_of_color(grid, RED):
        if comp.shape[0] == 1:
            singles.append(comp)
        else:
            assert comp.shape[0] == 2
            _domino_kind(comp)
            dominoes.append(comp)
    assert dominoes
    return singles, dominoes


def _validate_instance(inp: np.ndarray, expected: np.ndarray) -> None:
    assert inp.ndim == 2 and expected.ndim == 2
    assert max(inp.shape) <= 30 and max(expected.shape) <= 30

    frames = _validate_blue_frames(inp)
    singles, dominoes = _validate_red_components(inp)

    expected2 = main(inp)
    assert np.array_equal(expected2, expected)
    assert not np.array_equal(inp, expected), "transformation must do real work"

    naive = _naive_without_outline(inp)
    assert not np.array_equal(naive, expected), "outline step must be necessary"

    frame_counts = Counter(frame["cavity_hw"] for frame in frames)
    domino_counts = Counter(tuple(map(int, _outlined_patch_from_domino(comp).shape)) for comp in dominoes)
    for hw, count in domino_counts.items():
        assert frame_counts[hw] >= count
    assert any(hw not in MATCHED_CAVITIES.values() for hw in frame_counts) or any(len(s) == 1 for s in singles)

    out_red = _components_of_color(expected, RED)
    input_single_set = {tuple(map(int, comp[0])) for comp in singles}
    assert input_single_set <= {tuple(map(int, coord)) for comp in out_red for coord in comp}

    green_count = int(np.count_nonzero(expected == GREEN))
    assert green_count >= 10 * len(dominoes)


def _orientation_multiset(
    rng: np.random.Generator,
    domino_count: int,
) -> list[str]:
    kinds = [rng.choice(["H", "V"]) for _ in range(domino_count)]
    if domino_count >= 2 and rng.random() < 0.7 and len(set(kinds)) == 1:
        kinds[-1] = "V" if kinds[0] == "H" else "H"
    return [str(k) for k in kinds]


def _generation_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GRID_ROWS))
    w = int(rng.integers(*GRID_COLS))

    domino_count = int(rng.choice(DOMINO_COUNTS))
    kinds = _orientation_multiset(rng, domino_count)

    cavity_list = [MATCHED_CAVITIES[kind] for kind in kinds]
    if rng.random() < 0.6:
        cavity_list.append(DISTRACTOR_CAVITIES[int(rng.integers(0, len(DISTRACTOR_CAVITIES)))])

    if not (2 <= len(cavity_list) <= 4):
        return None

    frames = _sample_frame_layout(rng, h, w, cavity_list)
    if frames is None:
        return None

    matched_frames = [frame for frame in frames if frame["cavity_hw"] in MATCHED_CAVITIES.values()]
    distractor_frames = [frame for frame in frames if frame["cavity_hw"] not in MATCHED_CAVITIES.values()]
    if len(matched_frames) != domino_count:
        return None

    used = np.zeros((h, w), dtype=bool)
    for frame in frames:
        _mark_forbidden(used, frame["frame_box"], pad=1)

    dominoes: list[np.ndarray] = []
    for kind in kinds:
        placements = _enumerate_domino_positions(used, frames, kind)
        if not placements:
            return None
        comp = placements[int(rng.integers(0, len(placements)))]
        dominoes.append(comp)
        _mark_forbidden(used, (comp[:, 0].min() - 1, comp[:, 1].min() - 1, comp[:, 0].max() + 1, comp[:, 1].max() + 1), pad=1)

    single_count = int(rng.integers(0, MAX_SINGLES + 1))
    if not distractor_frames and single_count == 0:
        single_count = 1
    singles: list[tuple[int, int]] = []
    for _ in range(single_count):
        spots = _enumerate_single_positions(used)
        if not spots:
            break
        r, c = spots[int(rng.integers(0, len(spots)))]
        singles.append((r, c))
        _mark_forbidden(used, (r, c, r, c), pad=1)

    input_grid = _build_input_grid(h, w, frames, dominoes, singles)

    kind_counter = Counter(kinds)
    frame_kind_counter = Counter("H" if frame["cavity_hw"] == MATCHED_CAVITIES["H"] else "V" for frame in matched_frames)
    if kind_counter != frame_kind_counter:
        return None

    expected = np.zeros((h, w), dtype=int)
    for frame in frames:
        _draw_frame(expected, frame["frame_box"])
    matched_by_kind: dict[str, list[dict]] = {"H": [], "V": []}
    for frame in matched_frames:
        matched_by_kind["H" if frame["cavity_hw"] == MATCHED_CAVITIES["H"] else "V"].append(frame)
    for kind, comp in zip(kinds, dominoes):
        frame = matched_by_kind[kind].pop()
        cr0, cc0, _cr1, _cc1 = frame["cavity_box"]
        _paste(expected, _outlined_patch_from_domino(comp), cr0, cc0)
    for r, c in singles:
        expected[r, c] = RED

    if distractor_frames and rng.random() < 0.5:
        # Prefer samples where an empty cavity visibly remains.
        pass

    _validate_instance(input_grid, expected)
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _generation_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")


if __name__ == "__main__":
    sample = generate_input()
    result = main(sample)
    assert isinstance(sample, np.ndarray)
    assert isinstance(result, np.ndarray)
